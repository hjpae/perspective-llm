#!/usr/bin/env python3
"""
ATS08c — PersonaMem domain adaptation + held-out benchmark validation

Purpose
-------
Adapt the already-trained ATS06 CEAR recurrent human-state architecture to the
PersonaMem-v2 text domain WITHOUT changing the frozen sentence encoder or the
architecture, then evaluate on persona-disjoint benchmark users.

This is intentionally NOT a longitudinal-validity experiment. PersonaMem train/
val/benchmark provide persona-disjoint personalization data, while the final
stress probe is a controlled transition intervention.

Pipeline
--------
1) initialize from ATS06 seed-0 best checkpoint;
2) freeze the semantic encoder (precomputed embeddings), fine-tune ATS model;
3) select checkpoint on PersonaMem val users only;
4) evaluate benchmark users once: learned-g vs g=0;
5) run the same corrected 08b stable / one-off / sustained / recovery stress test
   on benchmark-native preference transitions.

The benchmark split is never used for training or checkpoint selection.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from sentence_transformers import SentenceTransformer

REPO_ROOT = Path(__file__).resolve().parents[2]
EPS = 1e-12

# Training protocol: domain adaptation, not architecture search.
TRAIN_SUPPORT_MIN = 4
TRAIN_SUPPORT_MAX = 12
TRAIN_QUERIES = 2
VAL_SUPPORT_FRAC = 0.67
MIN_QUERY = 2
USERS_PER_BATCH = 24
MAX_EPOCHS = 120
PATIENCE = 20
LR = 1e-4
WEIGHT_DECAY = 1e-4
SUPPORT_LOSS_WEIGHT = 0.25
G_NORM_WEIGHT = 1e-4
GRAD_CLIP = 1.0
VAL_PERMUTATIONS = 3
TEST_PERMUTATIONS = 8
BOOTSTRAP = 3000


@dataclass
class History:
    user_id: str
    items: np.ndarray
    choices: np.ndarray


@dataclass
class Episode:
    user_id: str
    support_items: np.ndarray
    support_choices: np.ndarray
    query_items: np.ndarray
    query_choices: np.ndarray


def resolve_path(x: str) -> Path:
    p = Path(x).expanduser()
    return p.resolve() if p.is_absolute() else (REPO_ROOT / p).resolve()


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def seed_all(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def det_seed(uid: str, seed: int, salt: int = 0) -> int:
    h = hashlib.sha1(f"{seed}|{salt}|{uid}".encode()).hexdigest()[:8]
    return int(h, 16)


def prepare_csv(A, B, csv_path: Path, split_name: str) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """Canonicalize PersonaMem MCQ rows into the ATS item format."""
    raw = pd.read_csv(csv_path, dtype={"persona_id": str})

    # ATS targets the user's own preference state. Forget requests are a separate
    # control problem and are excluded from this adaptation run.
    if "who" in raw.columns:
        raw = raw[raw["who"].fillna("self").astype(str).str.lower().isin(["self", ""])].copy()
    if "pref_type" in raw.columns:
        raw = raw[raw["pref_type"].fillna("").astype(str).str.lower() != "ask_to_forget"].copy()

    rows: List[Dict[str, Any]] = []
    items: Dict[str, Dict[str, Any]] = {}

    for _, r in raw.iterrows():
        item = B.canonical_item(A, r.to_dict())
        uid = B.clean(r.get("persona_id"))
        if not uid or item is None:
            continue

        hashes = [A.stable_hash(x) for x in item["candidates"]]
        item_id = A.stable_hash(A.stable_hash(item["prompt"]) + "|" + "|".join(hashes))
        rows.append({"user_id": uid, "item_id": item_id, "choice": int(item["choice"])})
        items.setdefault(item_id, {
            "item_id": item_id,
            "prompt": item["prompt"],
            "candidates": item["candidates"],
        })

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No usable rows in {csv_path}")

    # Prevent exact duplicate item leakage within a persona. Conflicting duplicate
    # labels are discarded; consistent duplicates collapse to one observation.
    cleaned = []
    for (_, _), g in df.groupby(["user_id", "item_id"], sort=False):
        ys = g["choice"].unique()
        if len(ys) == 1:
            cleaned.append(g.iloc[0])
    df = pd.DataFrame(cleaned).reset_index(drop=True)

    print(f"[{split_name}] rows={len(df):,} users={df.user_id.nunique():,} items={df.item_id.nunique():,}")
    return df, items


def build_embedding_cache(A, all_items: Dict[str, Dict[str, Any]], encoder_name: str,
                          cache_dir: Path, device: str, batch_size: int):
    cache_dir.mkdir(parents=True, exist_ok=True)
    epath = cache_dir / "pair_embeddings.npy"
    ipath = cache_dir / "item_index.json"
    mpath = cache_dir / "embedding_meta.json"

    item_ids = sorted(all_items)
    expected = {
        "encoder": encoder_name,
        "n_items": len(item_ids),
        "item_ids_sha1": hashlib.sha1("\n".join(item_ids).encode()).hexdigest(),
    }

    if epath.exists() and ipath.exists() and mpath.exists():
        meta = json.loads(mpath.read_text())
        if all(meta.get(k) == v for k, v in expected.items()):
            print("Loading cached PersonaMem frozen pair embeddings...")
            E = np.load(epath)
            item_to_idx = json.loads(ipath.read_text())
            return E, item_to_idx

    print(f"Encoding {len(item_ids):,} PersonaMem items with frozen {encoder_name}...")
    texts = []
    for iid in item_ids:
        x = all_items[iid]
        for c in x["candidates"]:
            texts.append(A.pair_text(x["prompt"], c))

    enc = SentenceTransformer(encoder_name, device=device)
    E2 = enc.encode(
        texts, batch_size=batch_size, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True,
    ).astype(np.float16)
    E = E2.reshape(len(item_ids), 4, -1)
    item_to_idx = {iid: i for i, iid in enumerate(item_ids)}

    np.save(epath, E)
    ipath.write_text(json.dumps(item_to_idx), encoding="utf-8")
    mpath.write_text(json.dumps({**expected, "embed_dim": int(E.shape[-1])}, indent=2), encoding="utf-8")
    return E, item_to_idx


class Store:
    def __init__(self, E: np.ndarray, device: str):
        self.E = torch.tensor(E.astype(np.float32), dtype=torch.float32, device=device)
        self.embed_dim = int(E.shape[-1])
    def get(self, idx):
        return self.E[torch.as_tensor(idx, dtype=torch.long, device=self.E.device)]


def make_histories(df: pd.DataFrame, item_to_idx: Dict[str, int]) -> List[History]:
    out = []
    for uid, g in df.groupby("user_id", sort=False):
        idx = [item_to_idx[x] for x in g.item_id]
        out.append(History(str(uid), np.asarray(idx, np.int64), g.choice.to_numpy(np.int64)))
    return out


def fixed_episodes(histories: List[History], seed: int) -> List[Episode]:
    out = []
    for h in histories:
        n = len(h.items)
        if n < TRAIN_SUPPORT_MIN + MIN_QUERY:
            continue
        rng = np.random.default_rng(det_seed(h.user_id, seed))
        idx = np.arange(n); rng.shuffle(idx)
        ns = max(TRAIN_SUPPORT_MIN, int(round(VAL_SUPPORT_FRAC * n)))
        ns = min(ns, n - MIN_QUERY)
        s, q = idx[:ns], idx[ns:]
        out.append(Episode(h.user_id, h.items[s], h.choices[s], h.items[q], h.choices[q]))
    return out


def sample_train_episode(h: History, rng: np.random.Generator):
    n = len(h.items)
    if n < TRAIN_SUPPORT_MIN + TRAIN_QUERIES:
        return None
    max_s = min(TRAIN_SUPPORT_MAX, n - TRAIN_QUERIES)
    ns = int(rng.integers(TRAIN_SUPPORT_MIN, max_s + 1))
    idx = rng.permutation(n)
    s, q = idx[:ns], idx[ns:ns + TRAIN_QUERIES]
    return h.items[s], h.choices[s], h.items[q], h.choices[q]


def make_batch(users: List[History], store: Store, rng: np.random.Generator, device: str):
    eps = [sample_train_episode(h, rng) for h in users]
    eps = [x for x in eps if x is not None]
    if not eps:
        return None
    B = len(eps); S = max(len(x[0]) for x in eps); Q = TRAIN_QUERIES
    si = np.zeros((B,S), np.int64); sy = np.zeros((B,S), np.int64); sm = np.zeros((B,S), bool)
    qi = np.zeros((B,Q), np.int64); qy = np.zeros((B,Q), np.int64)
    for b,(a,y,c,z) in enumerate(eps):
        si[b,:len(a)] = a; sy[b,:len(a)] = y; sm[b,:len(a)] = True
        qi[b] = c; qy[b] = z
    return {
        "support_z": store.get(si),
        "support_y": torch.tensor(sy, dtype=torch.long, device=device),
        "support_mask": torch.tensor(sm, dtype=torch.bool, device=device),
        "query_z": store.get(qi),
        "query_y": torch.tensor(qy, dtype=torch.long, device=device),
    }


def run_support(A, model, z, y, mask):
    B,S,_,_ = z.shape
    g = torch.zeros((B, A.G_DIM), device=z.device)
    err = A.initial_error_state(B, z.device)
    ce_sum = torch.tensor(0.0, device=z.device); n = 0
    for t in range(S):
        active = mask[:,t]
        if not active.any():
            continue
        logits, geo = model.predict(z[:,t], g)
        ce = F.cross_entropy(logits, y[:,t], reduction="none")
        ce_sum = ce_sum + ce[active].sum(); n += int(active.sum())
        ef, en = A.build_error_features(logits, y[:,t], err)
        gn, _ = model.update_g(g, z[:,t], geo, y[:,t], ef)
        g = torch.where(active[:,None], gn, g)
        err = type(err)(
            short=torch.where(active, en.short, err.short),
            long=torch.where(active, en.long, err.long),
        )
    return g, ce_sum / max(n,1)


def query_ce(A, model, z, y, g):
    B,Q,K,D = z.shape
    zz = z.reshape(B*Q,K,D); yy = y.reshape(B*Q)
    gg = g[:,None,:].expand(B,Q,A.G_DIM).reshape(B*Q,A.G_DIM)
    logits,_ = model.predict(zz,gg)
    return F.cross_entropy(logits,yy), logits


@torch.no_grad()
def infer_g(A, model, ep: Episode, store: Store, perm: np.ndarray):
    z = store.get(ep.support_items[perm]).unsqueeze(0)
    y = torch.tensor(ep.support_choices[perm], dtype=torch.long, device=store.E.device).unsqueeze(0)
    m = torch.ones((1,len(perm)), dtype=torch.bool, device=store.E.device)
    g,_ = run_support(A,model,z,y,m)
    return g[0]


@torch.no_grad()
def predict_queries(A, model, ep: Episode, store: Store, g: torch.Tensor):
    z = store.get(ep.query_items)
    gg = g.unsqueeze(0).expand(len(ep.query_items), A.G_DIM)
    logits,_ = model.predict(z,gg)
    return torch.softmax(logits,-1).cpu().numpy()


def prediction_rows(A, model, episodes: List[Episode], store: Store,
                    permutations: int, mode: str, seed: int):
    rows=[]
    for ep in episodes:
        if mode == "reset":
            P = predict_queries(A,model,ep,store,torch.zeros(A.G_DIM,device=store.E.device))
        else:
            P = np.zeros((len(ep.query_items),4), dtype=np.float64)
            for r in range(permutations):
                rng=np.random.default_rng(det_seed(ep.user_id,seed,salt=r))
                perm=rng.permutation(len(ep.support_items))
                g=infer_g(A,model,ep,store,perm)
                P += predict_queries(A,model,ep,store,g)/permutations
        pred=P.argmax(1)
        for j,y in enumerate(ep.query_choices):
            oh=np.zeros(4); oh[int(y)]=1
            rows.append({
                "user_id":ep.user_id,"query_index":j,"mode":mode,
                "correct":int(pred[j]==int(y)),
                "nll":float(-math.log(max(EPS,P[j,int(y)]))),
                "brier":float(np.square(P[j]-oh).sum()),
                "p_true":float(P[j,int(y)]),
            })
    return pd.DataFrame(rows)


def summarize(d: pd.DataFrame):
    return {
        "n_query":len(d),"n_users":d.user_id.nunique(),
        "accuracy":float(d.correct.mean()),"nll":float(d.nll.mean()),
        "brier":float(d.brier.mean()),"mean_p_true":float(d.p_true.mean()),
    }


def user_boot_delta(a: pd.DataFrame,b: pd.DataFrame,metric: str,seed: int,nboot:int=BOOTSTRAP):
    ua=a.groupby("user_id")[metric].mean(); ub=b.groupby("user_id")[metric].mean()
    common=ua.index.intersection(ub.index); delta=(ua.loc[common]-ub.loc[common]).to_numpy()
    rng=np.random.default_rng(seed); idx=rng.integers(0,len(delta),(nboot,len(delta)))
    means=delta[idx].mean(1)
    return {"metric":metric,"n_users":len(delta),"delta":float(delta.mean()),
            "ci_low":float(np.quantile(means,.025)),"ci_high":float(np.quantile(means,.975))}


def train(A, model, train_hist, val_eps, store, result_dir:Path, seed:int,
          max_epochs:int, patience:int, lr:float):
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=WEIGHT_DECAY)
    rng=np.random.default_rng(seed+100)
    best=float("inf"); best_state=None; best_epoch=None; bad=0; hist=[]

    for epoch in range(1,max_epochs+1):
        model.train(); order=np.arange(len(train_hist)); rng.shuffle(order)
        ls=[]; qs=[]; ss=[]
        for st in range(0,len(order),USERS_PER_BATCH):
            users=[train_hist[i] for i in order[st:st+USERS_PER_BATCH]]
            batch=make_batch(users,store,rng,store.E.device)
            if batch is None: continue
            g,sce=run_support(A,model,batch["support_z"],batch["support_y"],batch["support_mask"])
            qce,_=query_ce(A,model,batch["query_z"],batch["query_y"],g)
            loss=qce+SUPPORT_LOSS_WEIGHT*sce+G_NORM_WEIGHT*g.square().mean()
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),GRAD_CLIP); opt.step()
            ls.append(float(loss)); qs.append(float(qce)); ss.append(float(sce))

        model.eval()
        vd=prediction_rows(A,model,val_eps,store,VAL_PERMUTATIONS,"learned",seed+3000)
        vn=float(vd.nll.mean())
        row={"epoch":epoch,"train_loss":float(np.mean(ls)),"train_query_ce":float(np.mean(qs)),
             "train_support_ce":float(np.mean(ss)),"val_nll":vn,
             "person_scale":float(model.person_scale_log.exp().item())}
        hist.append(row)
        print(f"epoch {epoch:03d} | trainQ={row['train_query_ce']:.4f} trainS={row['train_support_ce']:.4f} val={vn:.4f} scale={row['person_scale']:.3f}")

        if vn < best - 1e-5:
            best=vn; best_epoch=epoch; bad=0
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            torch.save({"state_dict":best_state,"epoch":epoch,"val_nll":vn},result_dir/"best.pt")
        else:
            bad+=1
        pd.DataFrame(hist).to_csv(result_dir/"training_history.csv",index=False)
        if bad>=patience:
            print("early stopping"); break

    if best_state is None: raise RuntimeError("No best checkpoint")
    model.load_state_dict(best_state)
    return pd.DataFrame(hist),best_epoch,best


def bootstrap_mean(x,nboot,seed):
    x=np.asarray(x,float); x=x[np.isfinite(x)]
    rng=np.random.default_rng(seed); idx=rng.integers(0,len(x),(nboot,len(x))); m=x[idx].mean(1)
    return {"n":int(len(x)),"mean":float(x.mean()),"ci_low":float(np.quantile(m,.025)),"ci_high":float(np.quantile(m,.975))}


def run_corrected_stress(A,B,model,meta,pm_root:Path,result_dir:Path,device:str,seed:int,
                         pre_old:int=6,steps:int=8,recovery_old:int=4):
    """Same corrected geometry/readout logic as ATS08b, now using adapted model."""
    bench=pd.read_csv(pm_root/"benchmark/text/benchmark.csv",dtype={"persona_id":str})
    people=bench[["persona_id","raw_persona_file"]].dropna().drop_duplicates("persona_id")
    rng=np.random.default_rng(seed); cases=[]
    for i in rng.permutation(len(people)):
        r=people.iloc[int(i)]; pid=str(r.persona_id); remote=B.clean(r.raw_persona_file)
        raw=json.loads(B.find_raw(pm_root,remote).read_text(encoding="utf-8"))
        pairs=B.extract_usable_pairs(A,raw)
        if not pairs: continue
        pair=pairs[int(rng.integers(0,len(pairs)))]; pair["persona_id"]=pid; pair["raw_file"]=remote; cases.append(pair)
    print("stress usable benchmark personas:",len(cases))

    items=[]
    for c in cases:
        c["old_idx"]=len(items); items.append(c["old"])
        c["new_idx"]=len(items); items.append(c["new"])
    E=B.embed_items(A,items,meta["text_encoder"],device,128)

    pair_rows=[]; traj_rows=[]
    for ii,c in enumerate(cases,1):
        pid=c["persona_id"]; zo,zn=E[c["old_idx"]],E[c["new_idx"]]; yo,yn=c["old"]["choice"],c["new"]["choice"]
        g=torch.zeros(A.G_DIM,device=device); err=A.initial_error_state(1,device)
        for _ in range(pre_old): g,err,_=B.advance_one(A,model,zo,yo,g,err)
        gpre=g.clone(); epre=B.clone_error_state(err)

        maxstable=max(steps,recovery_old+1); stable={}; spn={}; spo={}; sa={}
        g=gpre.clone(); err=B.clone_error_state(epre)
        for t in range(1,maxstable+1):
            g,err,a=B.advance_one(A,model,zo,yo,g,err); stable[t]=g.clone();
            spn[t]=B.p_correct(model,zn,yn,g); spo[t]=B.p_correct(model,zo,yo,g); sa[t]=float(a.mean())

        change={}; cpn={}; cpo={}; ca={}; g=gpre.clone(); err=B.clone_error_state(epre)
        for t in range(1,steps+1):
            g,err,a=B.advance_one(A,model,zn,yn,g,err); change[t]=g.clone();
            cpn[t]=B.p_correct(model,zn,yn,g); cpo[t]=B.p_correct(model,zo,yo,g); ca[t]=float(a.mean())

        finalcontrast=change[steps]-stable[steps]
        for t in range(1,steps+1):
            traj_rows.append(B.make_traj_row(pid,"stable_old",t,t,stable[t],gpre,spn[t],spo[t],sa[t],stable[t],finalcontrast))
            traj_rows.append(B.make_traj_row(pid,"sustained_new",t,t,change[t],gpre,cpn[t],cpo[t],ca[t],stable[t],finalcontrast))

        g=gpre.clone(); err=B.clone_error_state(epre); g,err,a=B.advance_one(A,model,zn,yn,g,err)
        shockgap=B.l2(g,stable[1]); traj_rows.append(B.make_traj_row(pid,"shock_recovery",0,1,g,gpre,B.p_correct(model,zn,yn,g),B.p_correct(model,zo,yo,g),float(a.mean()),stable[1],finalcontrast))
        recgap={}
        for t in range(1,recovery_old+1):
            g,err,a=B.advance_one(A,model,zo,yo,g,err); matched=stable[t+1]; recgap[t]=B.l2(g,matched)
            traj_rows.append(B.make_traj_row(pid,"shock_recovery",t,t+1,g,gpre,B.p_correct(model,zn,yn,g),B.p_correct(model,zo,yo,g),float(a.mean()),matched,finalcontrast))

        gap1=B.l2(change[1],stable[1]); gapf=B.l2(change[steps],stable[steps]); rf=1-recgap[recovery_old]/shockgap if shockgap>EPS else np.nan
        pair_rows.append({
            "persona_id":pid,"old_pref":c["old_pref"],"new_pref":c["new_pref"],
            "change_gap_1":gap1,"change_gap_final":gapf,"sustained_minus_oneoff_gap":gapf-gap1,
            "semantic_new_effect_final":cpn[steps]-spn[steps],
            "semantic_old_effect_final":cpo[steps]-spo[steps],
            "alpha_new_minus_old_1":ca[1]-sa[1],
            "shock_gap_to_matched_stable":shockgap,
            "recovery_final_gap_to_matched_stable":recgap[recovery_old],
            "matched_recovery_fraction":rf,
        })
        if ii%25==0 or ii==len(cases): print(f"stress processed {ii}/{len(cases)}")

    P=pd.DataFrame(pair_rows); T=pd.DataFrame(traj_rows)
    P.to_csv(result_dir/"stress_pair_level_summary.csv",index=False)
    T.to_csv(result_dir/"stress_trajectories_full_g.csv",index=False)

    primary={}
    names=["change_gap_final","sustained_minus_oneoff_gap","semantic_new_effect_final","semantic_old_effect_final","alpha_new_minus_old_1","matched_recovery_fraction"]
    for j,k in enumerate(names): primary[k]=bootstrap_mean(P[k],BOOTSTRAP,seed+j+1)
    fractions={
        "final_change_gap_gt_oneoff_gap":float((P.change_gap_final>P.change_gap_1).mean()),
        "semantic_new_effect_positive":float((P.semantic_new_effect_final>0).mean()),
        "semantic_old_effect_negative":float((P.semantic_old_effect_final<0).mean()),
        "positive_matched_recovery":float((P.matched_recovery_fraction>0).mean()),
    }
    out={"probe_type":"PersonaMem-adapted model; held-out benchmark controlled stress",
         "n_personas":len(P),"primary_metrics":primary,"fractions":fractions,
         "interpretation_limit":"Controlled synthetic transition stress; not real longitudinal validation."}
    (result_dir/"stress_summary.json").write_text(json.dumps(out,indent=2),encoding="utf-8")
    return out


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--ats06-dir",required=True)
    p.add_argument("--ats06-source",default="ATS/models/ats_06_cear_g_ca.py")
    p.add_argument("--ats08b-source",default="ATS/models/ats_08b_personamem_corrected_geometry.py")
    p.add_argument("--personamem-root",default="data/personamem_v2")
    p.add_argument("--result-dir",required=True)
    p.add_argument("--device",default="cuda")
    p.add_argument("--seed",type=int,default=0)
    p.add_argument("--epochs",type=int,default=MAX_EPOCHS)
    p.add_argument("--patience",type=int,default=PATIENCE)
    p.add_argument("--lr",type=float,default=LR)
    p.add_argument("--embed-batch-size",type=int,default=128)
    args=p.parse_args(); seed_all(args.seed)

    atsdir=resolve_path(args.ats06_dir); pmroot=resolve_path(args.personamem_root); out=resolve_path(args.result_dir); out.mkdir(parents=True,exist_ok=True)
    A=load_module(resolve_path(args.ats06_source),"ats06_pm08c")
    B=load_module(resolve_path(args.ats08b_source),"ats08b_helpers08c")
    meta=json.loads((atsdir/"experiment_metadata.json").read_text())
    A.DEVICE=args.device; A.SEED=int(meta.get("seed",0)); A.G_DIM=int(meta["g_dim"]); A.COUPLER=meta["coupler"]; A.METRIC_RANK=int(meta["metric_rank"]); A.ALPHA_MIN=float(meta["alpha_min"]); A.ALPHA_MAX=float(meta["alpha_max"]); A.TEXT_ENCODER=meta["text_encoder"]

    print("="*88); print("ATS08c — PERSONAMEM DOMAIN ADAPTATION"); print("="*88)
    print("device:",args.device,"| ATS06 best epoch:",meta.get("best_epoch"),"| max epochs:",args.epochs)

    # PersonaMem's official train/val files are both development data and may
    # share persona_id. Benchmark is the untouched held-out persona set.
    # Build our own PERSON-DISJOINT train/val split from the union of train+val.
    src={}; all_items={}
    for split in ["train","val","benchmark"]:
        df,items=prepare_csv(
            A,B,pmroot/f"benchmark/text/{split}.csv",split
        )
        src[split]=df
        all_items.update(items)

    official_users={k:set(v.user_id.astype(str)) for k,v in src.items()}
    print(
        "official persona overlaps:",
        "train-val", len(official_users["train"] & official_users["val"]),
        "| train-benchmark", len(official_users["train"] & official_users["benchmark"]),
        "| val-benchmark", len(official_users["val"] & official_users["benchmark"]),
    )

    # Benchmark remains completely untouched. The released PersonaMem text
    # files contain a tiny persona overlap between development and benchmark,
    # so conservatively REMOVE those personas from development rather than
    # removing anything from benchmark.
    dev_users = official_users["train"] | official_users["val"]
    benchmark_users = official_users["benchmark"]
    overlap = dev_users & benchmark_users

    if overlap:
        print(
            "WARNING: removing benchmark-overlap personas from development:",
            sorted(overlap),
        )

    # Merge official train+val, excluding every benchmark persona.
    dev = pd.concat([src["train"], src["val"]], ignore_index=True)
    dev = dev[
        ~dev.user_id.astype(str).isin(benchmark_users)
    ].reset_index(drop=True)

    assert not (
        set(dev.user_id.astype(str)) & benchmark_users
    ), "benchmark leakage remains after filtering"

    cleaned=[]
    for (_, _), g in dev.groupby(["user_id","item_id"], sort=False):
        ys=g["choice"].unique()
        if len(ys)==1:
            cleaned.append(g.iloc[0])
    dev=pd.DataFrame(cleaned).reset_index(drop=True)

    # Select validation PERSONAS, not rows. Restrict selection to users with
    # enough observations to construct a fixed support/query episode.
    counts=dev.groupby("user_id").size()
    eligible_val_users=[
        str(uid) for uid,n in counts.items()
        if int(n) >= TRAIN_SUPPORT_MIN + MIN_QUERY
    ]

    # Deterministic seed-dependent ordering.
    eligible_val_users=sorted(
        eligible_val_users,
        key=lambda uid: hashlib.sha1(
            f"{args.seed}|pm08c-val|{uid}".encode()
        ).hexdigest(),
    )

    n_val=max(1, int(round(0.15 * len(eligible_val_users))))
    val_user_set=set(eligible_val_users[:n_val])

    dfs={
        "train": dev[~dev.user_id.astype(str).isin(val_user_set)]
                    .reset_index(drop=True),
        "val": dev[dev.user_id.astype(str).isin(val_user_set)]
                  .reset_index(drop=True),
        "benchmark": src["benchmark"].reset_index(drop=True),
    }

    users={k:set(v.user_id.astype(str)) for k,v in dfs.items()}

    assert not (users["train"] & users["val"])
    assert not (users["train"] & users["benchmark"])
    assert not (users["val"] & users["benchmark"])

    print(
        "custom person-disjoint split:",
        "train", len(users["train"]),
        "| val", len(users["val"]),
        "| benchmark", len(users["benchmark"]),
    )

    # Save exact split provenance.
    split_rows=[]
    for split_name in ["train","val","benchmark"]:
        for uid in sorted(users[split_name]):
            split_rows.append({
                "persona_id": uid,
                "split": split_name,
            })
    pd.DataFrame(split_rows).to_csv(
        out/"persona_split.csv", index=False
    )

    E,item_to_idx=build_embedding_cache(A,all_items,meta["text_encoder"],pmroot/"cache_ats08c",args.device,args.embed_batch_size)
    store=Store(E,args.device)
    train_hist=make_histories(dfs["train"],item_to_idx); val_hist=make_histories(dfs["val"],item_to_idx); test_hist=make_histories(dfs["benchmark"],item_to_idx)
    val_eps=fixed_episodes(val_hist,args.seed); test_eps=fixed_episodes(test_hist,args.seed)
    train_hist=[h for h in train_hist if len(h.items)>=TRAIN_SUPPORT_MIN+TRAIN_QUERIES]
    print("eligible users train/val/test:",len(train_hist),len(val_eps),len(test_eps))

    model=A.CEARHumanStateModel(store.embed_dim).to(args.device)
    ck=torch.load(atsdir/"best.pt",map_location="cpu",weights_only=False); model.load_state_dict(ck["state_dict"])
    nparams=sum(x.numel() for x in model.parameters() if x.requires_grad); print("trainable ATS params:",f"{nparams:,}","(semantic encoder frozen/precomputed)")

    hist,bepoch,bval=train(A,model,train_hist,val_eps,store,out,args.seed,args.epochs,args.patience,args.lr)
    print("best val epoch/NLL:",bepoch,bval)

    # One-shot benchmark evaluation after val checkpoint selection.
    model.eval()
    learned=prediction_rows(A,model,test_eps,store,TEST_PERMUTATIONS,"learned",args.seed+10000)
    reset=prediction_rows(A,model,test_eps,store,1,"reset",args.seed+20000)
    learned.to_csv(out/"benchmark_predictions_g_learned.csv",index=False); reset.to_csv(out/"benchmark_predictions_g_reset.csv",index=False)
    summ=[]
    for name,d in [("g_learned",learned),("g_reset_query_only",reset)]:
        x=summarize(d); x["model"]=name; summ.append(x)
    S=pd.DataFrame(summ).sort_values("nll"); S.to_csv(out/"benchmark_summary.csv",index=False)
    boots=[]
    for j,m in enumerate(["correct","nll","brier"]):
        r=user_boot_delta(learned,reset,m,args.seed+900+j); r["model_a"]="g_learned"; r["model_b"]="g_reset_query_only"; boots.append(r)
    pd.DataFrame(boots).to_csv(out/"benchmark_bootstrap_deltas.csv",index=False)

    stress=run_corrected_stress(A,B,model,meta,pmroot,out,args.device,args.seed)

    metadata={
        "ats06_source_dir":str(atsdir),"initial_checkpoint_epoch":meta.get("best_epoch"),
        "persona_splits_disjoint":True,"split_provenance":"custom 85/15 persona-disjoint split of official text train+val after removing any personas appearing in official benchmark; official benchmark untouched","train_users":len(users["train"]),"val_users":len(users["val"]),"benchmark_users":len(users["benchmark"]),
        "semantic_encoder":meta["text_encoder"],"semantic_encoder_frozen":True,
        "architecture_changed":False,"trainable_params":nparams,"best_epoch":bepoch,"best_val_nll":bval,
        "train_seed":args.seed,"max_epochs":args.epochs,"patience":args.patience,"lr":args.lr,
        "support_order_semantics":"unknown; randomized during adaptation and averaged at evaluation",
        "longitudinal_validity":False,
    }
    (out/"experiment_metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")

    print("\n"+"="*88); print("ATS08c — HELD-OUT BENCHMARK SUMMARY"); print("="*88); print(S.to_string(index=False))
    print("\nBootstrap learned-reset:"); print(pd.DataFrame(boots).to_string(index=False))
    print("\nStress primary metrics:")
    for k,v in stress["primary_metrics"].items(): print(f"{k:31s} {v['mean']:+.5f} 95%CI[{v['ci_low']:+.5f},{v['ci_high']:+.5f}]")
    print("\nWrote:",out)

if __name__=="__main__": main()
