"""
ATS / PersonaMem-v2 EDA
Run from the perspective-llm repo root.
Dependencies: pandas numpy matplotlib huggingface_hub
"""
from __future__ import annotations
import json, math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HF_REPO = "bowen-upenn/PersonaMem-v2"

def find_repo_root() -> Path:
    """Find perspective-llm root even when Spyder uses the script folder as --wdir."""
    here = Path(__file__).resolve()
    for candidate in [here.parent, *here.parents]:
        if ((candidate / "src" / "perspective_llm").exists()
                or (candidate / "environment.yml").exists()
                or (candidate / ".git").exists()):
            return candidate
    return Path.cwd()

REPO_ROOT = find_repo_root()
DATA_ROOT = REPO_ROOT / "data" / "personamem_v2"
RESULT_ROOT = REPO_ROOT / "results" / "ATS" / "eda" / "personamem_v2"
RAW_PERSONA_SAMPLE_N: Optional[int] = 120
CHRONOLOGY_AUDIT_N = 30
RANDOM_SEED = 0
SPLIT_FILES = {
    "train": "benchmark/text/train.csv",
    "val": "benchmark/text/val.csv",
    "test": "benchmark/text/benchmark.csv",
}
KNOWN_PERSONA_LIST_KEYS = [
    "stereotypical_preferences", "anti_stereotypical_preferences",
    "neutral_preferences", "therapy_background", "health_and_medical_conditions",
]

def hf_download(filename: str) -> Path:
    from huggingface_hub import hf_hub_download
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    return Path(hf_hub_download(repo_id=HF_REPO, repo_type="dataset",
                                filename=filename, local_dir=str(DATA_ROOT)))

def norm(x: Any) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    return str(x).strip()

def as_bool(x: Any) -> bool:
    return x if isinstance(x, bool) else norm(x).lower() in {"true","1","yes","y"}

def repo_rel(x: Any) -> str:
    s = norm(x).replace("\\", "/")
    return (s[s.index("data/"):] if "data/" in s else s).lstrip("./")

def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def canonical_message(msg):
    c = msg.get("content", "")
    if not isinstance(c, str):
        c = json.dumps(c, sort_keys=True, ensure_ascii=False)
    return norm(msg.get("role", "")).lower(), " ".join(c.split())

def find_subsequence(history, block):
    if not block or len(block) > len(history):
        return None
    H = [canonical_message(m) for m in history]
    B = [canonical_message(m) for m in block]
    n = len(B)
    for i in range(len(H)-n+1):
        if H[i:i+n] == B:
            return i
    return None

def load_splits():
    out = {}
    for split, rel in SPLIT_FILES.items():
        print(f"[benchmark] {split}: {rel}")
        df = pd.read_csv(hf_download(rel)); df["split"] = split; out[split] = df
    return out

def benchmark_summary(splits):
    rows=[]
    for split, df in splits.items():
        rows.append({
            "split":split, "rows":len(df), "personas":df.persona_id.nunique(),
            "rows_per_persona_mean":len(df)/df.persona_id.nunique(),
            "updated_true":int(df.updated.map(as_bool).sum()),
            "ask_to_forget":int((df.pref_type.fillna("")=="ask_to_forget").sum()),
            "who_self":int((df.who.fillna("").str.lower()=="self").sum()),
            "who_others":int((df.who.fillna("").str.lower()=="others").sum()),
        })
    return pd.DataFrame(rows)

def split_overlap(splits):
    sets={k:set(v.persona_id.astype(str).unique()) for k,v in splits.items()}; rows=[]
    names=list(sets)
    for i,a in enumerate(names):
        for b in names[i+1:]:
            ov=sets[a]&sets[b]
            rows.append({"split_a":a,"split_b":b,"n_overlap":len(ov),
                         "example_ids":",".join(sorted(ov)[:20])})
    return pd.DataFrame(rows)

def raw_refs(splits):
    frames=[]
    for split,df in splits.items():
        t=df[["persona_id","raw_persona_file","chat_history_32k_link"]].drop_duplicates().copy()
        t["split"]=split; frames.append(t)
    refs=pd.concat(frames,ignore_index=True).sort_values(["persona_id","split"]).drop_duplicates("persona_id")
    if RAW_PERSONA_SAMPLE_N is not None and RAW_PERSONA_SAMPLE_N < len(refs):
        refs=refs.sample(RAW_PERSONA_SAMPLE_N,random_state=RANDOM_SEED)
    return refs.sort_values("persona_id").reset_index(drop=True)

def persona_payload(raw, persona_id):
    pid=norm(persona_id)
    if pid in raw: return raw[pid]
    for k,v in raw.items():
        if norm(k)==pid: return v
    if len(raw)==1: return next(iter(raw.values()))
    raise KeyError(pid)

def extract_blocks(payload, pid):
    updates=payload.get("preference_updates",{})
    if not isinstance(updates,dict): updates={}
    out=[]
    for scenario,items in payload.get("conversations",{}).items():
        if not isinstance(items,list): continue
        for j,item in enumerate(items):
            if not isinstance(item,dict): continue
            pref=norm(item.get("preference")); prev=norm(item.get("prev_pref")); typ=norm(item.get("pref_type"))
            upd=as_bool(item.get("updated",False)); msgs=item.get("conversations",[])
            native=bool(upd and typ!="ask_to_forget" and prev and updates.get(prev)==pref)
            out.append({"persona_id":pid,"scenario":scenario,"local_idx":j,
                        "preference":pref,"prev_pref":prev,"pref_type":typ,
                        "who":norm(item.get("who")),"updated":upd,
                        "topic_preference":norm(item.get("topic_preference")),
                        "n_turns":len(msgs) if isinstance(msgs,list) else 0,
                        "has_qa":bool(norm(item.get("user_query")) and norm(item.get("correct_answer"))),
                        "is_native_update_pair":native,"_messages":msgs if isinstance(msgs,list) else []})
    return out

def prompt_keys(payload):
    keys=[]
    for k in payload.keys():
        if k=="stereotypical_preferences": break
        keys.append(k)
    return keys

def history_messages(obj):
    if isinstance(obj,list): return obj
    if isinstance(obj,dict):
        for k in ("chat_history","conversations","messages"):
            if isinstance(obj.get(k),list): return obj[k]
    raise ValueError("No history messages")

def run_raw_eda(splits):
    refs=raw_refs(splits); ps=[]; bs=[]; cs=[]; ex=[]
    print(f"\n[raw EDA] {len(refs)} unique personas")
    for n,(_,ref) in enumerate(refs.iterrows(),1):
        print(f"  {n:>3}/{len(refs)} persona={ref.persona_id}")
        try:
            rp=hf_download(repo_rel(ref.raw_persona_file)); payload=persona_payload(load_json(rp),ref.persona_id)
            blocks=extract_blocks(payload,ref.persona_id); updates=payload.get("preference_updates",{})
            row={"persona_id":ref.persona_id,"n_blocks":len(blocks),
                 "n_preference_updates_map":len(updates) if isinstance(updates,dict) else 0,
                 "n_updated_blocks":sum(b["updated"] for b in blocks),
                 "n_ask_to_forget_blocks":sum(b["pref_type"]=="ask_to_forget" for b in blocks),
                 "n_native_update_pairs":sum(b["is_native_update_pair"] for b in blocks),
                 "n_blocks_with_qa":sum(b["has_qa"] for b in blocks),
                 "n_self_blocks":sum(b["who"]=="self" for b in blocks),
                 "n_others_blocks":sum(b["who"]=="others" for b in blocks),
                 "mean_turns_per_block":float(np.mean([b["n_turns"] for b in blocks])) if blocks else np.nan,
                 "system_prompt_keys":"|".join(prompt_keys(payload))}
            for k in KNOWN_PERSONA_LIST_KEYS:
                v=payload.get(k,[]); row[f"n_{k}"]=len(v) if isinstance(v,list) else 0
            ps.append(row)
            for b in blocks: bs.append({k:v for k,v in b.items() if k!="_messages"})
            for b in blocks:
                if b["is_native_update_pair"] and len(ex)<100:
                    ex.append({"persona_id":ref.persona_id,"pref_type":b["pref_type"],"scenario":b["scenario"],
                               "old_preference":b["prev_pref"],"new_preference":b["preference"]})
            if n<=CHRONOLOGY_AUDIT_N:
                H=history_messages(load_json(hf_download(repo_rel(ref.chat_history_32k_link))))
                matched=[]
                for b in blocks:
                    start=find_subsequence(H,b["_messages"])
                    if start is not None: matched.append((start,b))
                matched.sort(key=lambda x:x[0]); ranks={(b["scenario"],b["local_idx"]):(r,s) for r,(s,b) in enumerate(matched)}
                for b in blocks:
                    rs=ranks.get((b["scenario"],b["local_idx"]))
                    cat="ask_to_forget" if b["pref_type"]=="ask_to_forget" else "native_update" if b["is_native_update_pair"] else "independent_or_other"
                    cs.append({"persona_id":ref.persona_id,"category":cat,"matched_in_32k":rs is not None,
                               "block_rank":None if rs is None else rs[0],"start_message_idx":None if rs is None else rs[1],
                               "normalized_rank":None if rs is None or len(matched)<=1 else rs[0]/(len(matched)-1),
                               "n_matched_blocks_persona":len(matched),"n_raw_blocks_persona":len(blocks)})
        except Exception as e:
            ps.append({"persona_id":ref.persona_id,"error":repr(e)})
    return pd.DataFrame(ps),pd.DataFrame(bs),pd.DataFrame(cs),pd.DataFrame(ex)

def save_hist(s, name, title, xlabel):
    s=pd.to_numeric(s,errors="coerce").dropna()
    if s.empty:return
    plt.figure(figsize=(8.5,5)); plt.hist(s,bins=30); plt.title(title); plt.xlabel(xlabel); plt.ylabel("Count")
    plt.tight_layout(); plt.savefig(RESULT_ROOT/name,dpi=160); plt.close()

def main():
    RESULT_ROOT.mkdir(parents=True,exist_ok=True)
    print("="*78); print("ATS / PersonaMem-v2 EDA"); print("="*78)
    splits=load_splits(); bsum=benchmark_summary(splits); ov=split_overlap(splits)
    print("\n[benchmark summary]"); print(bsum.to_string(index=False)); print("\n[persona overlap]"); print(ov.to_string(index=False))
    bsum.to_csv(RESULT_ROOT/"benchmark_split_summary.csv",index=False); ov.to_csv(RESULT_ROOT/"persona_overlap.csv",index=False)
    for col in ["pref_type","conversation_scenario","who","updated","sensitive_info","topic_query","topic_preference"]:
        frames=[]
        for split,df in splits.items():
            vc=df[col].fillna("<NA>").astype(str).value_counts().reset_index(); vc.columns=[col,"count"]; vc["fraction"]=vc["count"]/len(df); vc["split"]=split; frames.append(vc)
        pd.concat(frames,ignore_index=True).to_csv(RESULT_ROOT/f"benchmark_counts_{col}.csv",index=False)
    ps,blocks,chrono,examples=run_raw_eda(splits)
    ps.to_csv(RESULT_ROOT/"raw_persona_summary.csv",index=False); blocks.to_csv(RESULT_ROOT/"raw_block_table.csv",index=False)
    chrono.to_csv(RESULT_ROOT/"chronology_audit.csv",index=False); examples.to_csv(RESULT_ROOT/"native_update_examples.csv",index=False)
    if not blocks.empty:
        for col in ["scenario","pref_type","who","updated","has_qa"]:
            blocks[col].fillna("<NA>").astype(str).value_counts().rename_axis(col).reset_index(name="count").to_csv(RESULT_ROOT/f"raw_counts_{col}.csv",index=False)
    if not chrono.empty:
        csum=chrono.groupby("category").agg(n=("persona_id","size"),match_rate=("matched_in_32k","mean"),median_normalized_rank=("normalized_rank","median"),mean_normalized_rank=("normalized_rank","mean")).reset_index()
        csum.to_csv(RESULT_ROOT/"chronology_category_summary.csv",index=False); print("\n[chronology category summary]"); print(csum.to_string(index=False))
    if not ps.empty:
        save_hist(ps["n_blocks"],"raw_blocks_per_persona.png","Raw blocks per persona","Blocks")
        save_hist(ps["n_preference_updates_map"],"raw_updates_per_persona.png","Dataset-native updates per persona","Updates")
    key_counter=Counter()
    if "system_prompt_keys" in ps:
        for s in ps.system_prompt_keys.fillna(""):
            for k in s.split("|"):
                if k:key_counter[k]+=1
    pd.DataFrame(key_counter.most_common(),columns=["key","personas"]).to_csv(RESULT_ROOT/"system_prompt_key_frequency.csv",index=False)
    checks={"raw_personas_audited":int(len(ps)),"raw_blocks_audited":int(len(blocks)),
            "native_update_pairs":int(blocks.is_native_update_pair.sum()) if not blocks.empty else 0,
            "ask_to_forget_blocks":int((blocks.pref_type=="ask_to_forget").sum()) if not blocks.empty else 0,
            "chronology_personas_audited":int(chrono.persona_id.nunique()) if not chrono.empty else 0,
            "chronology_block_match_rate":float(chrono.matched_in_32k.mean()) if not chrono.empty else None}
    with open(RESULT_ROOT/"eda_checks.json","w",encoding="utf-8") as f: json.dump(checks,f,indent=2)
    print("\n[EDA checks]"); print(json.dumps(checks,indent=2)); print("\nWrote:",RESULT_ROOT.resolve())

if __name__=="__main__": main()
