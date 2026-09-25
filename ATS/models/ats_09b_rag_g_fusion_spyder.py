# -*- coding: utf-8 -*-
"""
ATS09b — Actual LLM/RAG + persistent ATS-g comparison
Spyder / Windows local edition

Primary question
----------------
Do the persistent ATS user state g and an actual LLM+RAG predictor carry
complementary predictive information about held-out PersonaMem preferences?

Primary fusion
--------------
A symmetric logarithmic opinion pool:

    P_fuse(y) ∝ P_ATS-g(y) ** (1-lambda) * P_RAG(y) ** lambda

lambda = 0 -> pure ATS-g
lambda = 1 -> pure RAG

To avoid selecting RAG budget k or lambda on the same users used to score them,
this script uses USER-DISJOINT CROSS-FITTING over the 196 benchmark personas:

  1. split users into N_FOLDS;
  2. for each held-out fold, choose RAG condition k* on the OTHER folds by NLL;
  3. choose lambda* on the OTHER folds by NLL;
  4. evaluate k*, lambda* only on the held-out users;
  5. concatenate all out-of-fold predictions;
  6. user-cluster bootstrap the paired differences.

This reuses the completed ATS09a gpt-5-mini predictions. No OpenAI API calls.
It loads the PersonaMem-adapted ATS08c checkpoint and computes ATS-g
probabilities on exactly the same 392 query IDs stored in the ATS09a manifest.

Recommended repository location
-------------------------------
    ATS/models/ats_09b_rag_g_fusion_spyder.py
"""

# %% IMPORTS
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd
import torch


# %% CONFIG — EDIT HERE IN SPYDER

DEVICE = "cpu"
SEED = 0

# Same permutation averaging used by ATS08c benchmark evaluation.
ATS_PERMUTATIONS = 8

# User-disjoint cross-fitting for k and lambda selection.
N_FOLDS = 5
LAMBDA_GRID = np.round(np.linspace(0.0, 1.0, 21), 10)
N_BOOTSTRAP = 5000
BOOTSTRAP_SEED = 20260925

# Actual LLM/RAG run to compare against.
# None -> auto-discover exact preferred directory below.
ATS09A_DIR: Optional[str] = None
PREFERRED_09A_DIR = "spyder_local_gpt-5-mini_q392_seed0"

# PersonaMem-adapted ATS checkpoint.
# None -> latest valid seed0_* directory.
ATS08C_DIR: Optional[str] = None

# Usually auto-resolved from ATS08c metadata.
ATS06_DIR: Optional[str] = None

ATS06_SOURCE = "ATS/models/ats_06_cear_g_ca.py"
ATS08B_SOURCE = "ATS/models/ats_08b_personamem_corrected_geometry.py"
ATS08C_SOURCE = "ATS/models/ats_08c_personamem_adaptation.py"
PERSONAMEM_ROOT = "data/personamem_v2"

# Dedicated benchmark pair-embedding cache. This uses the SAME frozen semantic
# encoder as ATS08c, but only benchmark items are encoded/indexed.
BENCHMARK_EMBED_CACHE = "data/personamem_v2/cache_ats09b_benchmark"

# Candidate actual-RAG history budgets. query_only is retained as a comparator,
# but is deliberately NOT eligible for k selection.
RAG_CANDIDATES = [
    "rag_1",
    "rag_3",
    "rag_5",
    "rag_10",
    "rag_all_support",
]

# Stable output directory; reruns overwrite deterministic CSV analysis outputs.
RESULT_LABEL = "gpt5mini_q392_crossfit"

EPS = 1e-12
PROB_COLS = ["p_A", "p_B", "p_C", "p_D"]


# %% PATHS / MODULE LOADING

def find_repo_root() -> Path:
    if "__file__" in globals():
        p = Path(__file__).resolve()
        for parent in [p.parent, *p.parents]:
            if (parent / "ATS").exists() and (parent / "README.md").exists():
                return parent

    cwd = Path.cwd().resolve()
    for parent in [cwd, *cwd.parents]:
        if (parent / "ATS").exists() and (parent / "README.md").exists():
            return parent

    raise RuntimeError(
        "Could not locate perspective-llm repository root. "
        "Put this file under ATS/models/ or launch Spyder from the repo."
    )


ROOT = find_repo_root()


def repo_path(x: str | Path) -> Path:
    p = Path(x).expanduser()
    return p.resolve() if p.is_absolute() else (ROOT / p).resolve()


def loadmod(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import module: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def latest_valid_dir(base: Path, pattern: str, required: Iterable[str]) -> Path:
    if not base.exists():
        raise FileNotFoundError(base)
    dirs = sorted(
        [p for p in base.glob(pattern) if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for d in dirs:
        if all((d / x).exists() for x in required):
            return d.resolve()
    raise FileNotFoundError(
        f"No valid result directory under {base}; required={list(required)}"
    )


def resolve_09a_dir() -> Path:
    if ATS09A_DIR:
        p = repo_path(ATS09A_DIR)
    else:
        preferred = ROOT / "results" / "ATS" / "rag_history_budget" / PREFERRED_09A_DIR
        if preferred.exists():
            p = preferred.resolve()
        else:
            base = ROOT / "results" / "ATS" / "rag_history_budget"
            matches = sorted(
                [d for d in base.glob("*gpt-5-mini*q392*seed0*") if d.is_dir()],
                key=lambda d: d.stat().st_mtime,
                reverse=True,
            )
            if not matches:
                raise FileNotFoundError(
                    "Could not find the completed gpt-5-mini q392 ATS09a result. "
                    "Set ATS09A_DIR explicitly."
                )
            p = matches[0].resolve()

    for name in ["predictions.csv", "query_retrieval_manifest.jsonl"]:
        if not (p / name).exists():
            raise FileNotFoundError(p / name)
    return p


def resolve_08c_dir() -> Path:
    if ATS08C_DIR:
        p = repo_path(ATS08C_DIR)
        for name in ["best.pt", "experiment_metadata.json"]:
            if not (p / name).exists():
                raise FileNotFoundError(p / name)
        return p

    return latest_valid_dir(
        ROOT / "results" / "ATS" / "personamem_adaptation",
        "seed0_*",
        ["best.pt", "experiment_metadata.json"],
    )


def resolve_06_dir(meta08: dict) -> Path:
    if ATS06_DIR:
        p = repo_path(ATS06_DIR)
        if not (p / "experiment_metadata.json").exists():
            raise FileNotFoundError(p / "experiment_metadata.json")
        return p

    # ATS08c metadata may contain a server path. Preserve its basename locally.
    src = str(meta08.get("ats06_source_dir", "")).strip()
    if src:
        candidate = (
            ROOT / "results" / "ATS" / "cear_g_ca" / Path(src).name
        ).resolve()
        if (candidate / "experiment_metadata.json").exists():
            return candidate

    return latest_valid_dir(
        ROOT / "results" / "ATS" / "cear_g_ca",
        "rep_seed0_*",
        ["experiment_metadata.json", "best.pt"],
    )


DIR09 = resolve_09a_dir()
DIR08 = resolve_08c_dir()
META08 = json.loads((DIR08 / "experiment_metadata.json").read_text(encoding="utf-8"))
DIR06 = resolve_06_dir(META08)
OUT = (ROOT / "results" / "ATS" / "rag_plus_g" / RESULT_LABEL).resolve()
OUT.mkdir(parents=True, exist_ok=True)

print("=" * 92)
print("ATS09b — ACTUAL RAG + PERSISTENT g — SPYDER LOCAL")
print("=" * 92)
print("repo root:    ", ROOT)
print("ATS09a dir:   ", DIR09)
print("ATS08c dir:   ", DIR08)
print("ATS06 dir:    ", DIR06)
print("device:       ", DEVICE)
print("folds:        ", N_FOLDS)
print("ATS perms:    ", ATS_PERMUTATIONS)
print("result dir:   ", OUT)


# %% LOAD ORIGINAL ATS MODULES / CONFIGURE ARCHITECTURE

A = loadmod(repo_path(ATS06_SOURCE), "ats06_09b")
B = loadmod(repo_path(ATS08B_SOURCE), "ats08b_09b")
C = loadmod(repo_path(ATS08C_SOURCE), "ats08c_09b")

META06 = json.loads((DIR06 / "experiment_metadata.json").read_text(encoding="utf-8"))

A.DEVICE = DEVICE
A.SEED = int(META06.get("seed", SEED))
A.G_DIM = int(META06["g_dim"])
A.COUPLER = str(META06["coupler"])
A.METRIC_RANK = int(META06["metric_rank"])
A.ALPHA_MIN = float(META06["alpha_min"])
A.ALPHA_MAX = float(META06["alpha_max"])
A.TEXT_ENCODER = str(META06["text_encoder"])

np.random.seed(SEED)
torch.manual_seed(SEED)


# %% LOAD ATS09a PREDICTIONS + EXACT QUERY MANIFEST

def read_jsonl(path: Path) -> list[dict]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


rag = pd.read_csv(DIR09 / "predictions.csv", dtype={"user_id": str, "query_id": str})
manifest = read_jsonl(DIR09 / "query_retrieval_manifest.jsonl")
man = pd.DataFrame(manifest)
man["user_id"] = man["user_id"].astype(str)
man["query_id"] = man["query_id"].astype(str)

required_cols = {
    "user_id", "query_id", "query_index", "true_choice", "condition", *PROB_COLS
}
missing_cols = required_cols - set(rag.columns)
if missing_cols:
    raise RuntimeError(f"ATS09a predictions missing columns: {sorted(missing_cols)}")

for c in PROB_COLS:
    rag[c] = rag[c].astype(float)

rag_conditions = set(rag["condition"].unique())
missing_conditions = [x for x in ["query_only", *RAG_CANDIDATES] if x not in rag_conditions]
if missing_conditions:
    raise RuntimeError(f"ATS09a is missing conditions: {missing_conditions}")

nq = man["query_id"].nunique()
nu = man["user_id"].nunique()
print(f"\nATS09a manifest: {nq} queries | {nu} users")
print("RAG conditions:", sorted(rag_conditions))

# Every condition must cover the exact same selected queries.
expected_ids = set(man["query_id"])
for cond, g in rag.groupby("condition"):
    ids = set(g["query_id"])
    if ids != expected_ids:
        raise RuntimeError(
            f"Condition {cond!r} does not cover the exact ATS09a manifest: "
            f"missing={len(expected_ids - ids)}, extra={len(ids - expected_ids)}"
        )


# %% RECONSTRUCT EXACT ATS08c BENCHMARK EPISODES + EMBEDDINGS

pm = repo_path(PERSONAMEM_ROOT)
bench_csv = pm / "benchmark" / "text" / "benchmark.csv"
if not bench_csv.exists():
    raise FileNotFoundError(bench_csv)

bench_df, bench_items = C.prepare_csv(A, B, bench_csv, "benchmark")
item_ids = sorted(bench_items)
item_to_idx_expected = {iid: i for i, iid in enumerate(item_ids)}

histories = C.make_histories(bench_df, item_to_idx_expected)
episodes = C.fixed_episodes(histories, SEED)
ep_by_user = {str(ep.user_id): ep for ep in episodes}

# Dedicated benchmark cache; same encoder, same normalized pair-text embeddings.
E, item_to_idx = C.build_embedding_cache(
    A,
    bench_items,
    META08["semantic_encoder"],
    repo_path(BENCHMARK_EMBED_CACHE),
    DEVICE,
    128,
)
store = C.Store(E, DEVICE)

# Strong invariant: ATS09a and ATS09b must share the same benchmark item indexing.
if item_to_idx != item_to_idx_expected:
    raise RuntimeError("Benchmark item index mismatch between ATS09a reconstruction and ATS09b cache.")

# Verify every manifest record against the reconstructed fixed episode.
for r in manifest:
    uid = str(r["user_id"])
    if uid not in ep_by_user:
        raise RuntimeError(f"Manifest user not in ATS08c fixed benchmark episodes: {uid}")
    ep = ep_by_user[uid]
    qj = int(r["query_index"])
    qidx = int(r["query_item_idx"])
    y = int(r["true_choice"])
    if qj >= len(ep.query_items):
        raise RuntimeError(f"query_index out of range for user {uid}: {qj}")
    if int(ep.query_items[qj]) != qidx:
        raise RuntimeError(
            f"Query item mismatch for {uid}/{qj}: "
            f"episode={int(ep.query_items[qj])}, manifest={qidx}"
        )
    if int(ep.query_choices[qj]) != y:
        raise RuntimeError(
            f"True-choice mismatch for {uid}/{qj}: "
            f"episode={int(ep.query_choices[qj])}, manifest={y}"
        )

print("Exact ATS09a ↔ ATS08c query alignment: PASS")


# %% LOAD PERSONAMEM-ADAPTED ATS08c MODEL

model = A.CEARHumanStateModel(store.embed_dim).to(DEVICE)
ck = torch.load(DIR08 / "best.pt", map_location="cpu", weights_only=False)
model.load_state_dict(ck["state_dict"])
model.eval()
for p in model.parameters():
    p.requires_grad_(False)

print("ATS08c checkpoint epoch:", ck.get("epoch", META08.get("best_epoch")))
print("ATS08c checkpoint val NLL:", ck.get("val_nll", META08.get("best_val_nll")))
print("g dimension:", A.G_DIM)


# %% COMPUTE ATS-g + ATS-reset ON EXACTLY THE 392 MANIFEST QUERIES

@torch.no_grad()
def ats_probs_for_manifest() -> pd.DataFrame:
    selected_by_user: dict[str, list[dict]] = {}
    for r in manifest:
        selected_by_user.setdefault(str(r["user_id"]), []).append(r)

    rows = []
    users = sorted(selected_by_user)

    for ui, uid in enumerate(users, 1):
        ep = ep_by_user[uid]

        # g=0 query-only ATS prediction.
        g0 = torch.zeros(A.G_DIM, device=DEVICE)
        P0 = C.predict_queries(A, model, ep, store, g0)

        # Own persistent g, averaged over the same kind of random support-order
        # permutations used in ATS08c benchmark evaluation.
        Pg = np.zeros_like(P0, dtype=np.float64)
        for rep in range(ATS_PERMUTATIONS):
            rng = np.random.default_rng(
                C.det_seed(uid, SEED + 10000, salt=rep)
            )
            perm = rng.permutation(len(ep.support_items))
            g = C.infer_g(A, model, ep, store, perm)
            Pg += C.predict_queries(A, model, ep, store, g) / float(ATS_PERMUTATIONS)

        for r in selected_by_user[uid]:
            qj = int(r["query_index"])
            y = int(r["true_choice"])
            qid = str(r["query_id"])

            for name, P in [("ats_reset", P0), ("ats_g", Pg)]:
                p = np.asarray(P[qj], dtype=float)
                p = p / p.sum()
                oh = np.zeros(4, dtype=float)
                oh[y] = 1.0
                row = {
                    "user_id": uid,
                    "query_id": qid,
                    "query_index": qj,
                    "true_choice": y,
                    "condition": name,
                    "pred_choice": int(np.argmax(p)),
                    "correct": int(np.argmax(p) == y),
                    "p_true": float(p[y]),
                    "nll": float(-math.log(max(EPS, p[y]))),
                    "brier": float(np.square(p - oh).sum()),
                }
                row.update({PROB_COLS[j]: float(p[j]) for j in range(4)})
                rows.append(row)

        if ui % 25 == 0 or ui == len(users):
            print(f"ATS probabilities: {ui}/{len(users)} users")

    return pd.DataFrame(rows)


ats = ats_probs_for_manifest()
ats.to_csv(OUT / "ats_predictions_exact_manifest.csv", index=False)


# %% NORMALIZE ATS09a TABLE TO COMMON METRIC NAMES

def normalize_rag_predictions(d: pd.DataFrame) -> pd.DataFrame:
    x = d.copy()
    x["user_id"] = x["user_id"].astype(str)
    x["query_id"] = x["query_id"].astype(str)
    x["nll"] = x["elicited_nll"].astype(float)
    x["brier"] = x["elicited_brier"].astype(float)
    return x


rag = normalize_rag_predictions(rag)

# Query-level key/label alignment across all systems.
key_cols = ["user_id", "query_id", "query_index", "true_choice"]
base_keys = man[key_cols].copy()
base_keys["user_id"] = base_keys["user_id"].astype(str)
base_keys["query_id"] = base_keys["query_id"].astype(str)

for name, d in [("ATS", ats), ("RAG", rag)]:
    check = d[key_cols].drop_duplicates()
    merged = base_keys.merge(check, on=key_cols, how="outer", indicator=True)
    if not (merged["_merge"] == "both").all():
        raise RuntimeError(f"{name} query/label alignment failed.")


# %% FUSION / METRICS

def probs(df: pd.DataFrame) -> np.ndarray:
    return df[PROB_COLS].to_numpy(dtype=float)


def fuse_arrays(Pg: np.ndarray, Pr: np.ndarray, lam: float) -> np.ndarray:
    """Symmetric log-opinion pool. lam=0 ATS-g; lam=1 RAG."""
    lp = (
        (1.0 - float(lam)) * np.log(np.clip(Pg, EPS, 1.0))
        + float(lam) * np.log(np.clip(Pr, EPS, 1.0))
    )
    lp -= lp.max(axis=1, keepdims=True)
    P = np.exp(lp)
    return P / P.sum(axis=1, keepdims=True)


def frame_from_probs(keys: pd.DataFrame, P: np.ndarray, condition: str, **extra) -> pd.DataFrame:
    out = keys.copy().reset_index(drop=True)
    Y = out["true_choice"].to_numpy(dtype=int)
    pred = P.argmax(axis=1)
    oh = np.eye(4, dtype=float)[Y]
    out["condition"] = condition
    out["pred_choice"] = pred
    out["correct"] = (pred == Y).astype(int)
    out["p_true"] = P[np.arange(len(P)), Y]
    out["nll"] = -np.log(np.clip(out["p_true"].to_numpy(float), EPS, 1.0))
    out["brier"] = np.square(P - oh).sum(axis=1)
    for j, c in enumerate(PROB_COLS):
        out[c] = P[:, j]
    for k, v in extra.items():
        out[k] = v
    return out


def mean_user_nll(d: pd.DataFrame) -> float:
    return float(d.groupby("user_id")["nll"].mean().mean())


def summarize(d: pd.DataFrame, condition: str) -> dict:
    return {
        "condition": condition,
        "n_query": int(len(d)),
        "n_users": int(d["user_id"].nunique()),
        "accuracy": float(d["correct"].mean()),
        "nll": float(d["nll"].mean()),
        "brier": float(d["brier"].mean()),
        "mean_p_true": float(d["p_true"].mean()),
    }


def paired_bootstrap(a: pd.DataFrame, b: pd.DataFrame, metric: str,
                     name_a: str, name_b: str, seed: int) -> dict:
    col = "correct" if metric == "accuracy" else metric
    ua = a.groupby("user_id")[col].mean()
    ub = b.groupby("user_id")[col].mean()
    common = ua.index.intersection(ub.index)
    delta = (ua.loc[common] - ub.loc[common]).to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(delta), size=(N_BOOTSTRAP, len(delta)))
    boot = delta[idx].mean(axis=1)
    return {
        "model_a": name_a,
        "model_b": name_b,
        "metric": metric,
        "n_users": int(len(delta)),
        "delta": float(delta.mean()),
        "ci_low": float(np.quantile(boot, 0.025)),
        "ci_high": float(np.quantile(boot, 0.975)),
    }


# %% USER-DISJOINT FOLDS

users = sorted(man["user_id"].astype(str).unique())
rng = np.random.default_rng(SEED + 9000)
rng.shuffle(users)
fold_of = {uid: i % N_FOLDS for i, uid in enumerate(users)}

fold_assignments = pd.DataFrame(
    [{"user_id": uid, "fold": fold_of[uid]} for uid in sorted(fold_of)]
)
fold_assignments.to_csv(OUT / "crossfit_user_folds.csv", index=False)


# %% CROSS-FITTED RAG-k + LAMBDA SELECTION

ats_g = ats[ats["condition"] == "ats_g"].copy()
ats_reset = ats[ats["condition"] == "ats_reset"].copy()
query_only = rag[rag["condition"] == "query_only"].copy()

rag_by_cond = {
    c: rag[rag["condition"] == c].copy()
    for c in RAG_CANDIDATES
}

# Convenience lookup: exact ordering by keys for safe fusion.
def aligned_pair(ats_df: pd.DataFrame, rag_df: pd.DataFrame, users_subset: set[str]):
    cols = key_cols + PROB_COLS
    a = ats_df[ats_df.user_id.isin(users_subset)][cols].copy()
    r = rag_df[rag_df.user_id.isin(users_subset)][cols].copy()
    m = a.merge(r, on=key_cols, suffixes=("_g", "_r"), validate="one_to_one")
    Pg = m[[f"{c}_g" for c in PROB_COLS]].to_numpy(float)
    Pr = m[[f"{c}_r" for c in PROB_COLS]].to_numpy(float)
    keys = m[key_cols].copy()
    return keys, Pg, Pr


fold_k_rows = []
fold_lambda_rows = []
fold_choice_rows = []
oof_rag_parts = []
oof_fuse_parts = []

# Also build OOF sensitivity curves: for each lambda, use the fold-selected k,
# but do NOT optimize lambda. Useful to see whether complementarity is broad or
# depends on one narrow tuned value.
sensitivity_parts: dict[float, list[pd.DataFrame]] = {
    float(l): [] for l in LAMBDA_GRID
}

for fold in range(N_FOLDS):
    test_users = {u for u, f in fold_of.items() if f == fold}
    train_users = set(users) - test_users

    # --------------------------------------------------------------
    # Step 1: select actual-RAG history budget k* on training users.
    # --------------------------------------------------------------
    k_scores = []
    for cond in RAG_CANDIDATES:
        d = rag_by_cond[cond]
        tr = d[d.user_id.isin(train_users)]
        score = mean_user_nll(tr)
        k_scores.append((score, cond))
        fold_k_rows.append({
            "fold": fold,
            "rag_condition": cond,
            "train_user_mean_nll": score,
            "n_train_users": int(tr.user_id.nunique()),
        })

    k_scores.sort(key=lambda x: (x[0], x[1]))
    best_k_nll, best_cond = k_scores[0]
    best_rag = rag_by_cond[best_cond]

    # --------------------------------------------------------------
    # Step 2: select lambda* on training users, same selected k*.
    # --------------------------------------------------------------
    tr_keys, Pg_tr, Pr_tr = aligned_pair(ats_g, best_rag, train_users)
    lambda_scores = []

    for lam in LAMBDA_GRID:
        Pf = fuse_arrays(Pg_tr, Pr_tr, float(lam))
        fd = frame_from_probs(tr_keys, Pf, "fusion_train")
        score = mean_user_nll(fd)
        lambda_scores.append((score, float(lam)))
        fold_lambda_rows.append({
            "fold": fold,
            "rag_condition": best_cond,
            "lambda_rag": float(lam),
            "train_user_mean_nll": score,
            "n_train_users": int(fd.user_id.nunique()),
        })

    lambda_scores.sort(key=lambda x: (x[0], x[1]))
    best_lam_nll, best_lam = lambda_scores[0]

    fold_choice_rows.append({
        "fold": fold,
        "n_train_users": len(train_users),
        "n_test_users": len(test_users),
        "selected_rag_condition": best_cond,
        "selected_rag_train_user_mean_nll": best_k_nll,
        "selected_lambda_rag": best_lam,
        "selected_fusion_train_user_mean_nll": best_lam_nll,
    })

    # --------------------------------------------------------------
    # Step 3: held-out fold only.
    # --------------------------------------------------------------
    rag_test = best_rag[best_rag.user_id.isin(test_users)].copy()
    rag_test["condition"] = "rag_selected_cf"
    rag_test["selected_rag_condition"] = best_cond
    rag_test["selected_lambda_rag"] = best_lam
    rag_test["fold"] = fold
    oof_rag_parts.append(rag_test)

    te_keys, Pg_te, Pr_te = aligned_pair(ats_g, best_rag, test_users)
    Pf_te = fuse_arrays(Pg_te, Pr_te, best_lam)
    fuse_test = frame_from_probs(
        te_keys,
        Pf_te,
        "ats_g_plus_rag_cf",
        selected_rag_condition=best_cond,
        selected_lambda_rag=best_lam,
        fold=fold,
    )
    oof_fuse_parts.append(fuse_test)

    # Fixed-lambda OOF sensitivity, using only fold-selected k.
    for lam in LAMBDA_GRID:
        P_lam = fuse_arrays(Pg_te, Pr_te, float(lam))
        sensitivity_parts[float(lam)].append(
            frame_from_probs(
                te_keys,
                P_lam,
                "fusion_sensitivity",
                lambda_rag=float(lam),
                selected_rag_condition=best_cond,
                fold=fold,
            )
        )

    print(
        f"fold {fold}: k*={best_cond:15s} | lambda*={best_lam:.2f} | "
        f"train NLL rag={best_k_nll:.4f}, fusion={best_lam_nll:.4f}"
    )


fold_k_df = pd.DataFrame(fold_k_rows)
fold_lambda_df = pd.DataFrame(fold_lambda_rows)
fold_choices = pd.DataFrame(fold_choice_rows)
fold_k_df.to_csv(OUT / "crossfit_k_selection_surface.csv", index=False)
fold_lambda_df.to_csv(OUT / "crossfit_lambda_selection_surface.csv", index=False)
fold_choices.to_csv(OUT / "crossfit_selected_hyperparameters.csv", index=False)

rag_cf = pd.concat(oof_rag_parts, ignore_index=True)
fuse_cf = pd.concat(oof_fuse_parts, ignore_index=True)

# Exact one-row-per-query checks.
for name, d in [("rag_cf", rag_cf), ("fusion_cf", fuse_cf)]:
    if len(d) != nq or d["query_id"].nunique() != nq:
        raise RuntimeError(f"{name} does not contain exactly the {nq} manifest queries.")

rag_cf.to_csv(OUT / "predictions_rag_selected_crossfit.csv", index=False)
fuse_cf.to_csv(OUT / "predictions_ats_g_plus_rag_crossfit.csv", index=False)


# %% OOF LAMBDA SENSITIVITY

sensitivity_rows = []
for lam in LAMBDA_GRID:
    d = pd.concat(sensitivity_parts[float(lam)], ignore_index=True)
    s = summarize(d, f"lambda_{float(lam):.2f}")
    s["lambda_rag"] = float(lam)
    sensitivity_rows.append(s)

lambda_sensitivity = pd.DataFrame(sensitivity_rows).sort_values("lambda_rag")
lambda_sensitivity.to_csv(OUT / "lambda_sensitivity_oof.csv", index=False)


# %% MAIN SUMMARIES

# Rename/retain only columns needed for common summaries.
conditions: Dict[str, pd.DataFrame] = {
    "llm_query_only": query_only,
    "ats_reset": ats_reset,
    "ats_g": ats_g,
    "rag_selected_cf": rag_cf,
    "ats_g_plus_rag_cf": fuse_cf,
}

summary = pd.DataFrame([
    summarize(d, name)
    for name, d in conditions.items()
]).sort_values("nll")
summary.to_csv(OUT / "main_summary.csv", index=False)

# Descriptive ATS09a budget curve copied into the same analysis directory.
rag_budget_summary = pd.DataFrame([
    summarize(rag[rag.condition == c], c)
    for c in ["query_only", *RAG_CANDIDATES]
])
rag_budget_summary.to_csv(OUT / "rag_budget_summary_recomputed.csv", index=False)


# %% USER-CLUSTER PAIRED BOOTSTRAPS

comparisons = [
    # Main complementarity test.
    ("ats_g_plus_rag_cf", "rag_selected_cf"),
    ("ats_g_plus_rag_cf", "ats_g"),

    # Standalone representation comparison.
    ("ats_g", "rag_selected_cf"),

    # Context/history baselines.
    ("rag_selected_cf", "llm_query_only"),
    ("ats_g", "ats_reset"),
]

boot_rows = []
seed_offset = 0
for a, b in comparisons:
    for metric in ["accuracy", "nll", "brier"]:
        boot_rows.append(
            paired_bootstrap(
                conditions[a],
                conditions[b],
                metric,
                a,
                b,
                BOOTSTRAP_SEED + seed_offset,
            )
        )
        seed_offset += 1

boots = pd.DataFrame(boot_rows)
boots.to_csv(OUT / "bootstrap_deltas.csv", index=False)


# %% OPTIONAL DIAGNOSTICS: EXTREME CONFIDENCE / NLL TAILS

diag_rows = []
for name, d in conditions.items():
    ptrue = d["p_true"].to_numpy(float)
    nll = d["nll"].to_numpy(float)
    diag_rows.append({
        "condition": name,
        "median_p_true": float(np.median(ptrue)),
        "p_true_lt_0.01": float(np.mean(ptrue < 0.01)),
        "p_true_lt_0.05": float(np.mean(ptrue < 0.05)),
        "median_nll": float(np.median(nll)),
        "nll_p95": float(np.quantile(nll, 0.95)),
        "nll_max": float(np.max(nll)),
    })

pd.DataFrame(diag_rows).to_csv(OUT / "confidence_tail_diagnostics.csv", index=False)


# %% METADATA

metadata = {
    "experiment": "ATS09b actual LLM/RAG + persistent ATS-g",
    "primary_fusion": (
        "symmetric logarithmic opinion pool: "
        "P_fuse proportional to P_ats_g^(1-lambda) * P_rag^lambda"
    ),
    "lambda_interpretation": {
        "0": "pure ATS-g",
        "1": "pure actual LLM/RAG",
    },
    "selection_protocol": (
        f"{N_FOLDS}-fold user-disjoint cross-fitting on the benchmark cohort; "
        "for each held-out fold, RAG budget and lambda are chosen by mean per-user "
        "NLL on the other folds only"
    ),
    "important_scope": (
        "This is an out-of-fold cross-validated estimate on benchmark users, not a "
        "single pristine benchmark evaluation with hyperparameters selected on an "
        "external validation LLM/RAG run."
    ),
    "ats09a_dir": str(DIR09),
    "ats08c_dir": str(DIR08),
    "ats06_dir": str(DIR06),
    "device": DEVICE,
    "n_queries": int(nq),
    "n_users": int(nu),
    "n_folds": N_FOLDS,
    "rag_candidates": RAG_CANDIDATES,
    "lambda_grid": [float(x) for x in LAMBDA_GRID],
    "ats_permutations": ATS_PERMUTATIONS,
    "bootstrap": N_BOOTSTRAP,
    "semantic_encoder": META08["semantic_encoder"],
    "g_dim": int(A.G_DIM),
    "api_calls_in_09b": False,
}
(OUT / "experiment_metadata.json").write_text(
    json.dumps(metadata, indent=2, ensure_ascii=False),
    encoding="utf-8",
)


# %% CONSOLE

print("\n" + "=" * 92)
print("ATS09b — MAIN SUMMARY")
print("=" * 92)
print(summary.to_string(index=False))

print("\n[cross-fitted selections]")
print(fold_choices.to_string(index=False))

print("\n[paired user-cluster bootstrap: model_a - model_b]")
print(boots.to_string(index=False))

print("\n[OOF lambda sensitivity]")
print(lambda_sensitivity.to_string(index=False))

print("\nInterpretation guide:")
print("  accuracy: positive delta for model_a - model_b is better")
print("  NLL/Brier: negative delta for model_a - model_b is better")
print("  Primary complementarity test:")
print("    ats_g_plus_rag_cf vs rag_selected_cf")
print("  If fusion improves held-out RAG with lambda* strictly between 0 and 1,")
print("  ATS-g carries predictive information not exhausted by retrieved history.")
print("  If lambda* repeatedly goes to 0, ATS-g dominates; if it goes to 1, RAG dominates.")
print("\nWrote:", OUT)
