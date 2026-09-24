#!/usr/bin/env python3
"""
ATS08b — PersonaMem-v2 corrected geometry stress probe (NO TRAINING)

This is the corrected follow-up to ATS08a. It reuses the frozen ATS06 seed-0
checkpoint and fixes the recovery/control geometry.

Core corrections
----------------
1. Every-step full g vector is saved.
2. CHANGE is compared with a matched STABLE trajectory at the same elapsed update.
3. Recovery is measured against the matched STABLE trajectory, not against g_pre.
4. Semantic change is measured as CHANGE minus matched STABLE on the same query.

Interpretation remains cross-dataset transfer only: ATS06 was trained on Community
Alignment, not PersonaMem-v2. Weak semantic transfer is not a falsification of ATS.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

REPO_ROOT = Path(__file__).resolve().parents[2]
EPS = 1e-12


def clean(x: Any) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return " ".join(str(x).strip().split())


def norm(x: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", clean(x).lower()).strip()


def as_bool(x: Any) -> bool:
    return x is True or clean(x).lower() in {"1", "true", "t", "yes", "y"}


def parse_list(x: Any) -> List[str]:
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    for fn in (json.loads, ast.literal_eval):
        try:
            y = fn(str(x))
            if isinstance(y, (list, tuple)):
                return [clean(v) for v in y]
        except Exception:
            pass
    return []


def parse_query(x: Any) -> str:
    if isinstance(x, dict):
        return clean(x.get("content", x))
    s = clean(x)
    for fn in (json.loads, ast.literal_eval):
        try:
            y = fn(s)
            if isinstance(y, dict):
                return clean(y.get("content", y))
        except Exception:
            pass
    return s


def recursive_records(x: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(x, dict):
        if {"preference", "user_query", "correct_answer", "incorrect_answers"}.issubset(x):
            yield x
        for v in x.values():
            yield from recursive_records(v)
    elif isinstance(x, list):
        for v in x:
            yield from recursive_records(v)


def resolve_repo_path(x: str) -> Path:
    p = Path(x).expanduser()
    return p.resolve() if p.is_absolute() else (REPO_ROOT / p).resolve()


def load_module_from_path(path: Path):
    spec = importlib.util.spec_from_file_location("ats06_pmstress08b", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import ATS06 source: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def canonical_item(A, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    prompt = A.normalize_text(parse_query(row.get("user_query")))
    correct = A.normalize_text(row.get("correct_answer"))
    wrong = [A.normalize_text(x) for x in parse_list(row.get("incorrect_answers"))]

    if not prompt or not correct or len(wrong) != 3 or not all(wrong):
        return None

    tagged = [(correct, 1)] + [(x, 0) for x in wrong]
    hashed = [(A.stable_hash(text), text, y) for text, y in tagged]
    if len(set(h for h, _, _ in hashed)) != 4:
        return None

    ordered = sorted(hashed, key=lambda z: z[0])
    candidates = [text for _, text, _ in ordered]
    labels = [y for _, _, y in ordered]

    return {
        "prompt": prompt,
        "candidates": candidates,
        "choice": int(labels.index(1)),
    }


def find_raw(root: Path, remote: str) -> Path:
    candidates = [
        root / remote,
        root / Path(remote).name,
        root / "data/raw_data" / Path(remote).name,
    ]
    for p in candidates:
        if p.exists():
            return p
    hits = list(root.rglob(Path(remote).name))
    if hits:
        return hits[0]
    raise FileNotFoundError(remote)


def extract_usable_pairs(A, raw_obj: Any) -> List[Dict[str, Any]]:
    rows = list(recursive_records(raw_obj))
    by_pref: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        pref = clean(r.get("preference"))
        if pref:
            by_pref.setdefault(norm(pref), []).append(r)

    pairs: List[Dict[str, Any]] = []
    for new_r in rows:
        if not as_bool(new_r.get("updated")):
            continue
        if clean(new_r.get("pref_type")).lower() == "ask_to_forget":
            continue

        old_pref = clean(new_r.get("prev_pref"))
        new_pref = clean(new_r.get("preference"))
        if not old_pref or not new_pref or norm(old_pref) == norm(new_pref):
            continue

        new_item = canonical_item(A, new_r)
        if new_item is None:
            continue

        old_candidates = []
        for old_r in by_pref.get(norm(old_pref), []):
            old_item = canonical_item(A, old_r)
            if old_item is not None:
                old_candidates.append((as_bool(old_r.get("updated")), old_item))
        if not old_candidates:
            continue

        old_candidates.sort(key=lambda x: x[0])  # prefer original/non-updated record
        pairs.append({
            "old_pref": old_pref,
            "new_pref": new_pref,
            "old": old_candidates[0][1],
            "new": new_item,
        })

    return pairs


def embed_items(A, items: List[Dict[str, Any]], encoder_name: str,
                device: str, batch_size: int) -> List[torch.Tensor]:
    texts: List[str] = []
    for item in items:
        for candidate in item["candidates"]:
            texts.append(A.pair_text(item["prompt"], candidate))

    encoder = SentenceTransformer(encoder_name, device=device)
    E = encoder.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    E = E.reshape(len(items), 4, -1)
    return [torch.tensor(E[i], dtype=torch.float32, device=device) for i in range(len(items))]


def clone_error_state(e):
    return type(e)(short=e.short.clone(), long=e.long.clone())


@torch.no_grad()
def advance_one(A, model, z: torch.Tensor, y: int,
                g: torch.Tensor, err):
    zb = z.unsqueeze(0)
    gb = g.unsqueeze(0)
    yb = torch.tensor([int(y)], dtype=torch.long, device=z.device)
    logits, geo = model.predict(zb, gb)
    err_feat, new_err = A.build_error_features(logits, yb, err)
    g_new, alpha = model.update_g(gb, zb, geo, yb, err_feat)
    return g_new.squeeze(0), new_err, alpha.squeeze(0)


@torch.no_grad()
def p_correct(model, z: torch.Tensor, y: int, g: torch.Tensor) -> float:
    logits, _ = model.predict(z.unsqueeze(0), g.unsqueeze(0))
    p = torch.softmax(logits.squeeze(0), dim=-1)
    return float(p[int(y)].item())


def l2(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(a - b).item())


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    den = float(torch.linalg.vector_norm(a).item() * torch.linalg.vector_norm(b).item())
    if den < EPS:
        return float("nan")
    return float(torch.dot(a, b).item() / den)


def projection(delta: torch.Tensor, direction: torch.Tensor) -> float:
    den = torch.linalg.vector_norm(direction)
    if float(den.item()) < EPS:
        return float("nan")
    return float(torch.dot(delta, direction / den).item())


def bootstrap_mean_ci(x, n_boot: int, seed: int) -> Dict[str, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {"n": 0, "mean": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    return {
        "n": int(len(x)),
        "mean": float(x.mean()),
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
    }


def g_columns(g: torch.Tensor) -> Dict[str, float]:
    arr = g.detach().cpu().numpy().astype(float)
    return {f"g_{i:02d}": float(v) for i, v in enumerate(arr)}


def make_traj_row(pid: str, condition: str, step: int, elapsed: int,
                  g: torch.Tensor, g_pre: torch.Tensor,
                  p_new: float, p_old: float, alpha: float,
                  matched_stable: Optional[torch.Tensor] = None,
                  final_contrast: Optional[torch.Tensor] = None) -> Dict[str, Any]:
    row = {
        "persona_id": pid,
        "condition": condition,
        "step": int(step),
        "elapsed_updates": int(elapsed),
        "g_disp_pre": l2(g, g_pre),
        "g_cos_pre": cosine(g, g_pre),
        "p_new": float(p_new),
        "p_old": float(p_old),
        "alpha": float(alpha),
    }
    if matched_stable is not None:
        delta = g - matched_stable
        row["g_gap_matched_stable"] = l2(g, matched_stable)
        row["contrast_projection"] = (
            projection(delta, final_contrast) if final_contrast is not None else np.nan
        )
    else:
        row["g_gap_matched_stable"] = np.nan
        row["contrast_projection"] = np.nan
    row.update(g_columns(g))
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ats06-dir", required=True)
    p.add_argument("--ats06-source", default="ATS/models/ats_06_cear_g_ca.py")
    p.add_argument("--personamem-root", default="data/personamem_v2")
    p.add_argument("--result-dir", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pre-old", type=int, default=6)
    p.add_argument("--stress-steps", type=int, default=8)
    p.add_argument("--recovery-old", type=int, default=4)
    p.add_argument("--bootstrap", type=int, default=3000)
    p.add_argument("--embed-batch-size", type=int, default=128)
    args = p.parse_args()

    ats06_dir = resolve_repo_path(args.ats06_dir)
    ats06_source = resolve_repo_path(args.ats06_source)
    pm_root = resolve_repo_path(args.personamem_root)
    result_dir = resolve_repo_path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    meta = json.loads((ats06_dir / "experiment_metadata.json").read_text(encoding="utf-8"))
    A = load_module_from_path(ats06_source)
    A.DEVICE = args.device
    A.SEED = int(meta.get("seed", 0))
    A.G_DIM = int(meta["g_dim"])
    A.COUPLER = str(meta["coupler"])
    A.METRIC_RANK = int(meta["metric_rank"])
    A.ALPHA_MIN = float(meta["alpha_min"])
    A.ALPHA_MAX = float(meta["alpha_max"])
    A.TEXT_ENCODER = str(meta["text_encoder"])

    model = A.CEARHumanStateModel(int(meta["semantic_embed_dim"])).to(args.device)
    ckpt = torch.load(ats06_dir / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    for q in model.parameters():
        q.requires_grad_(False)

    print("=" * 88)
    print("ATS08b — CORRECTED PERSONAMEM GEOMETRY STRESS PROBE (NO TRAINING)")
    print("=" * 88)
    print("ATS06 best epoch:", meta.get("best_epoch"))
    print("DEVICE:          ", args.device)
    print("pre_old:         ", args.pre_old)
    print("stress_steps:    ", args.stress_steps)
    print("recovery_old:    ", args.recovery_old)

    benchmark_path = pm_root / "benchmark/text/benchmark.csv"
    if not benchmark_path.exists():
        raise FileNotFoundError(benchmark_path)
    benchmark = pd.read_csv(benchmark_path, dtype={"persona_id": str})
    people = benchmark[["persona_id", "raw_persona_file"]].dropna().drop_duplicates("persona_id")

    rng = np.random.default_rng(args.seed)
    cases: List[Dict[str, Any]] = []
    for i in rng.permutation(len(people)):
        r = people.iloc[int(i)]
        pid = str(r["persona_id"])
        remote = clean(r["raw_persona_file"])
        raw = json.loads(find_raw(pm_root, remote).read_text(encoding="utf-8"))
        pairs = extract_usable_pairs(A, raw)
        if not pairs:
            continue
        pair = pairs[int(rng.integers(0, len(pairs)))]
        pair["persona_id"] = pid
        pair["raw_file"] = remote
        cases.append(pair)

    print("usable benchmark personas:", len(cases))
    if len(cases) < 20:
        raise RuntimeError("Too few usable benchmark personas")

    manifest = pd.DataFrame([{
        "persona_id": c["persona_id"],
        "old_pref": c["old_pref"],
        "new_pref": c["new_pref"],
        "raw_file": c["raw_file"],
    } for c in cases])
    manifest.to_csv(result_dir / "pair_manifest.csv", index=False)

    items: List[Dict[str, Any]] = []
    for c in cases:
        c["old_idx"] = len(items); items.append(c["old"])
        c["new_idx"] = len(items); items.append(c["new"])

    E = embed_items(A, items, meta["text_encoder"], args.device, args.embed_batch_size)

    pair_rows: List[Dict[str, Any]] = []
    traj_rows: List[Dict[str, Any]] = []

    for ii, c in enumerate(cases, 1):
        pid = c["persona_id"]
        zo, zn = E[c["old_idx"]], E[c["new_idx"]]
        yo, yn = c["old"]["choice"], c["new"]["choice"]

        g = torch.zeros(A.G_DIM, device=args.device)
        err = A.initial_error_state(1, args.device)
        for _ in range(args.pre_old):
            g, err, _ = advance_one(A, model, zo, yo, g, err)
        g_pre = g.clone()
        err_pre = clone_error_state(err)

        p_new_pre = p_correct(model, zn, yn, g_pre)
        p_old_pre = p_correct(model, zo, yo, g_pre)
        traj_rows.append(make_traj_row(
            pid, "pre", 0, 0, g_pre, g_pre, p_new_pre, p_old_pre, np.nan
        ))

        # STABLE trajectory: old evidence continues. Need enough states to match recovery.
        max_stable_steps = max(args.stress_steps, args.recovery_old + 1)
        stable_states: Dict[int, torch.Tensor] = {}
        stable_pnew: Dict[int, float] = {}
        stable_pold: Dict[int, float] = {}
        stable_alpha: Dict[int, float] = {}
        g = g_pre.clone(); err = clone_error_state(err_pre)
        for t in range(1, max_stable_steps + 1):
            g, err, a = advance_one(A, model, zo, yo, g, err)
            stable_states[t] = g.clone()
            stable_pnew[t] = p_correct(model, zn, yn, g)
            stable_pold[t] = p_correct(model, zo, yo, g)
            stable_alpha[t] = float(a.mean().item())

        # CHANGE trajectory: repeated new evidence.
        change_states: Dict[int, torch.Tensor] = {}
        change_pnew: Dict[int, float] = {}
        change_pold: Dict[int, float] = {}
        change_alpha: Dict[int, float] = {}
        g = g_pre.clone(); err = clone_error_state(err_pre)
        for t in range(1, args.stress_steps + 1):
            g, err, a = advance_one(A, model, zn, yn, g, err)
            change_states[t] = g.clone()
            change_pnew[t] = p_correct(model, zn, yn, g)
            change_pold[t] = p_correct(model, zo, yo, g)
            change_alpha[t] = float(a.mean().item())

        # Final change-specific contrast, generic stable drift removed.
        final_contrast = change_states[args.stress_steps] - stable_states[args.stress_steps]

        # Save stable and change trajectories now that final contrast is known.
        for t in range(1, args.stress_steps + 1):
            traj_rows.append(make_traj_row(
                pid, "stable_old", t, t,
                stable_states[t], g_pre,
                stable_pnew[t], stable_pold[t], stable_alpha[t],
                matched_stable=stable_states[t],
                final_contrast=final_contrast,
            ))
            traj_rows.append(make_traj_row(
                pid, "sustained_new", t, t,
                change_states[t], g_pre,
                change_pnew[t], change_pold[t], change_alpha[t],
                matched_stable=stable_states[t],
                final_contrast=final_contrast,
            ))

        # SHOCK + RECOVERY. At elapsed t+1, compare to stable old^(t+1).
        g = g_pre.clone(); err = clone_error_state(err_pre)
        g, err, a = advance_one(A, model, zn, yn, g, err)
        shock_state = g.clone()
        shock_gap = l2(shock_state, stable_states[1])
        traj_rows.append(make_traj_row(
            pid, "shock_recovery", 0, 1,
            shock_state, g_pre,
            p_correct(model, zn, yn, shock_state),
            p_correct(model, zo, yo, shock_state),
            float(a.mean().item()),
            matched_stable=stable_states[1],
            final_contrast=final_contrast,
        ))

        recovery_gap_by_step: Dict[int, float] = {}
        for t in range(1, args.recovery_old + 1):
            g, err, a = advance_one(A, model, zo, yo, g, err)
            elapsed = t + 1
            matched = stable_states[elapsed]
            gap = l2(g, matched)
            recovery_gap_by_step[t] = gap
            traj_rows.append(make_traj_row(
                pid, "shock_recovery", t, elapsed,
                g, g_pre,
                p_correct(model, zn, yn, g),
                p_correct(model, zo, yo, g),
                float(a.mean().item()),
                matched_stable=matched,
                final_contrast=final_contrast,
            ))

        recovery_final_gap = recovery_gap_by_step[args.recovery_old]
        matched_recovery_fraction = (
            1.0 - recovery_final_gap / shock_gap if shock_gap > EPS else np.nan
        )

        gap1 = l2(change_states[1], stable_states[1])
        gap_final = l2(change_states[args.stress_steps], stable_states[args.stress_steps])
        proj1 = projection(change_states[1] - stable_states[1], final_contrast)
        proj_final = projection(final_contrast, final_contrast)

        pair_rows.append({
            "persona_id": pid,
            "old_pref": c["old_pref"],
            "new_pref": c["new_pref"],
            "p_new_pre": p_new_pre,
            "p_old_pre": p_old_pre,
            "stable_final_disp_pre": l2(stable_states[args.stress_steps], g_pre),
            "change_final_disp_pre": l2(change_states[args.stress_steps], g_pre),
            "change_gap_1": gap1,
            "change_gap_final": gap_final,
            "sustained_minus_oneoff_gap": gap_final - gap1,
            "change_projection_1": proj1,
            "change_projection_final": proj_final,
            "semantic_new_effect_final": (
                change_pnew[args.stress_steps] - stable_pnew[args.stress_steps]
            ),
            "semantic_old_effect_final": (
                change_pold[args.stress_steps] - stable_pold[args.stress_steps]
            ),
            "p_new_change_final": change_pnew[args.stress_steps],
            "p_new_stable_final": stable_pnew[args.stress_steps],
            "p_old_change_final": change_pold[args.stress_steps],
            "p_old_stable_final": stable_pold[args.stress_steps],
            "alpha_new_1": change_alpha[1],
            "alpha_old_1": stable_alpha[1],
            "alpha_new_minus_old_1": change_alpha[1] - stable_alpha[1],
            "shock_gap_to_matched_stable": shock_gap,
            "recovery_final_gap_to_matched_stable": recovery_final_gap,
            "matched_recovery_fraction": matched_recovery_fraction,
        })

        if ii % 25 == 0 or ii == len(cases):
            print(f"processed {ii}/{len(cases)}")

    pair_df = pd.DataFrame(pair_rows)
    traj_df = pd.DataFrame(traj_rows)
    pair_df.to_csv(result_dir / "pair_level_summary.csv", index=False)
    traj_df.to_csv(result_dir / "trajectories_full_g.csv", index=False)

    # Dose summary: matched CHANGE-vs-STABLE gap and semantic effect per step.
    dose_rows: List[Dict[str, Any]] = []
    for t in range(1, args.stress_steps + 1):
        change_t = traj_df[(traj_df.condition == "sustained_new") & (traj_df.step == t)]
        stable_t = traj_df[(traj_df.condition == "stable_old") & (traj_df.step == t)]
        c = change_t.set_index("persona_id")
        s = stable_t.set_index("persona_id")
        common = c.index.intersection(s.index)
        gap = c.loc[common, "g_gap_matched_stable"].to_numpy()
        pnew_eff = (c.loc[common, "p_new"] - s.loc[common, "p_new"]).to_numpy()
        pold_eff = (c.loc[common, "p_old"] - s.loc[common, "p_old"]).to_numpy()
        alpha_eff = (c.loc[common, "alpha"] - s.loc[common, "alpha"]).to_numpy()
        proj = c.loc[common, "contrast_projection"].to_numpy()
        for j, (name, arr) in enumerate([
            ("matched_change_gap", gap),
            ("semantic_new_effect", pnew_eff),
            ("semantic_old_effect", pold_eff),
            ("alpha_new_minus_old", alpha_eff),
            ("contrast_projection", proj),
        ]):
            b = bootstrap_mean_ci(arr, args.bootstrap, args.seed + 1000*t + j)
            dose_rows.append({"step": t, "metric": name, **b})
    pd.DataFrame(dose_rows).to_csv(result_dir / "dose_summary.csv", index=False)

    # Recovery summary by recovery step, always compared to matched stable elapsed time.
    recovery_rows: List[Dict[str, Any]] = []
    shock = traj_df[(traj_df.condition == "shock_recovery") & (traj_df.step == 0)]
    shock_map = shock.set_index("persona_id")["g_gap_matched_stable"]
    for t in range(0, args.recovery_old + 1):
        rr = traj_df[(traj_df.condition == "shock_recovery") & (traj_df.step == t)].copy()
        rr = rr[rr.persona_id.isin(shock_map.index)]
        denom = rr.persona_id.map(shock_map).to_numpy(dtype=float)
        gap = rr["g_gap_matched_stable"].to_numpy(dtype=float)
        frac = np.where(denom > EPS, 1.0 - gap / denom, np.nan)
        b_gap = bootstrap_mean_ci(gap, args.bootstrap, args.seed + 5000 + t)
        b_frac = bootstrap_mean_ci(frac, args.bootstrap, args.seed + 6000 + t)
        recovery_rows.append({"recovery_step": t, "metric": "gap_to_matched_stable", **b_gap})
        recovery_rows.append({"recovery_step": t, "metric": "matched_recovery_fraction", **b_frac})
    pd.DataFrame(recovery_rows).to_csv(result_dir / "recovery_summary.csv", index=False)

    primary_metrics = {}
    metric_names = [
        "change_gap_final",
        "sustained_minus_oneoff_gap",
        "semantic_new_effect_final",
        "semantic_old_effect_final",
        "alpha_new_minus_old_1",
        "matched_recovery_fraction",
    ]
    for j, name in enumerate(metric_names):
        primary_metrics[name] = bootstrap_mean_ci(
            pair_df[name].to_numpy(), args.bootstrap, args.seed + j + 1
        )

    fractions = {
        "final_change_gap_gt_oneoff_gap": float(
            (pair_df.change_gap_final > pair_df.change_gap_1).mean()
        ),
        "semantic_new_effect_positive": float(
            (pair_df.semantic_new_effect_final > 0).mean()
        ),
        "semantic_old_effect_negative": float(
            (pair_df.semantic_old_effect_final < 0).mean()
        ),
        "positive_matched_recovery": float(
            (pair_df.matched_recovery_fraction > 0).mean()
        ),
    }

    out = {
        "probe_type": "corrected zero-training CA-to-PersonaMem cross-dataset transfer",
        "n_personas": int(len(pair_df)),
        "ats06_best_epoch": meta.get("best_epoch"),
        "seed": args.seed,
        "protocol": {
            "pre_old": args.pre_old,
            "stress_steps": args.stress_steps,
            "recovery_old": args.recovery_old,
        },
        "primary_metrics": primary_metrics,
        "fractions": fractions,
        "correction": (
            "Recovery is measured against the stable-old trajectory at the same elapsed update; "
            "change effects are measured relative to matched stable-old states."
        ),
        "interpretation_limit": (
            "This is zero-shot cross-dataset transfer. ATS06 was trained on Community Alignment, "
            "not PersonaMem-v2."
        ),
    }
    (result_dir / "stress_summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("\n" + "=" * 88)
    print("ATS08b CORRECTED STRESS SUMMARY")
    print("=" * 88)
    for k, v in primary_metrics.items():
        print(f"{k:31s} {v['mean']:+.5f} 95%CI[{v['ci_low']:+.5f},{v['ci_high']:+.5f}]")
    print("\nFractions:")
    for k, v in fractions.items():
        print(f"  {k:34s} {v:.3f}")
    print("\nDesired corrected signature:")
    print("  change_gap_final > 0")
    print("  sustained_minus_oneoff_gap > 0")
    print("  semantic_new_effect_final > 0")
    print("  matched_recovery_fraction > 0")
    print("\nWrote:", result_dir)


if __name__ == "__main__":
    main()
