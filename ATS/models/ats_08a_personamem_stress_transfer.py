#!/usr/bin/env python3
from __future__ import annotations

import argparse, ast, importlib.util, json, re, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

REPO = Path(__file__).resolve().parents[2]
EPS = 1e-12


def clean(x):
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return " ".join(str(x).strip().split())


def norm(x):
    return re.sub(r"[^a-z0-9]+", " ", clean(x).lower()).strip()


def as_bool(x):
    return x is True or clean(x).lower() in {"1", "true", "t", "yes", "y"}


def parse_list(x):
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


def parse_query(x):
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


def walk(x):
    if isinstance(x, dict):
        if {"preference", "user_query", "correct_answer",
            "incorrect_answers"}.issubset(x):
            yield x
        for v in x.values():
            yield from walk(v)
    elif isinstance(x, list):
        for v in x:
            yield from walk(v)


def load_module(path):
    spec = importlib.util.spec_from_file_location("ats06_pmstress", str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def canonical_item(A, r):
    prompt = A.normalize_text(parse_query(r.get("user_query")))
    correct = A.normalize_text(r.get("correct_answer"))
    wrong = [A.normalize_text(x) for x in parse_list(r.get("incorrect_answers"))]

    if not prompt or not correct or len(wrong) != 3 or not all(wrong):
        return None

    tagged = [(correct, 1)] + [(x, 0) for x in wrong]
    hashed = [(A.stable_hash(x), x, y) for x, y in tagged]

    if len(set(h for h, _, _ in hashed)) != 4:
        return None

    ordered = sorted(hashed, key=lambda z: z[0])
    candidates = [x for _, x, _ in ordered]
    labels = [y for _, _, y in ordered]

    return {
        "prompt": prompt,
        "candidates": candidates,
        "choice": labels.index(1),
    }


def find_raw(root, remote):
    for p in [
        root / remote,
        root / Path(remote).name,
        root / "data/raw_data" / Path(remote).name,
    ]:
        if p.exists():
            return p

    hits = list(root.rglob(Path(remote).name))
    if hits:
        return hits[0]

    raise FileNotFoundError(remote)


def usable_pairs(A, raw):
    rows = list(walk(raw))

    by_pref = {}
    for r in rows:
        p = clean(r.get("preference"))
        if p:
            by_pref.setdefault(norm(p), []).append(r)

    pairs = []

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

        old_items = []
        for old_r in by_pref.get(norm(old_pref), []):
            q = canonical_item(A, old_r)
            if q is not None:
                old_items.append(q)

        if not old_items:
            continue

        pairs.append({
            "old_pref": old_pref,
            "new_pref": new_pref,
            "old": old_items[0],
            "new": new_item,
        })

    return pairs


def embed_items(A, items, encoder_name, device):
    texts = []

    for x in items:
        for c in x["candidates"]:
            texts.append(A.pair_text(x["prompt"], c))

    enc = SentenceTransformer(encoder_name, device=device)

    z = enc.encode(
        texts,
        batch_size=128,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    z = z.reshape(len(items), 4, -1)

    return [
        torch.tensor(z[i], dtype=torch.float32, device=device)
        for i in range(len(items))
    ]


def clone_err(e):
    return type(e)(short=e.short.clone(), long=e.long.clone())


@torch.no_grad()
def advance(A, model, z, y, g, err):
    zb = z.unsqueeze(0)
    gb = g.unsqueeze(0)
    yb = torch.tensor([int(y)], dtype=torch.long, device=z.device)

    logits, geo = model.predict(zb, gb)
    err_feat, new_err = A.build_error_features(logits, yb, err)
    g_new, alpha = model.update_g(gb, zb, geo, yb, err_feat)

    return g_new.squeeze(0), new_err, float(alpha.mean().item())


@torch.no_grad()
def p_correct(model, z, y, g):
    logits, _ = model.predict(z.unsqueeze(0), g.unsqueeze(0))
    p = torch.softmax(logits.squeeze(0), dim=-1)
    return float(p[int(y)].item())


def l2(a, b):
    return float(torch.linalg.vector_norm(a - b).item())


def cosine(a, b):
    den = float(torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b))
    if den < EPS:
        return np.nan
    return float(torch.dot(a, b) / den)


def bootstrap(x, n=3000, seed=0):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n, len(x)))
    means = x[idx].mean(axis=1)

    return {
        "n": len(x),
        "mean": float(x.mean()),
        "ci_low": float(np.quantile(means, .025)),
        "ci_high": float(np.quantile(means, .975)),
    }


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--ats06-dir", required=True)
    ap.add_argument("--result-dir", required=True)
    ap.add_argument("--personamem-root",
                    default="data/personamem_v2")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()

    root = REPO / args.personamem_root
    result = REPO / args.result_dir
    result.mkdir(parents=True, exist_ok=True)

    atsdir = REPO / args.ats06_dir

    meta = json.loads(
        (atsdir / "experiment_metadata.json").read_text()
    )

    A = load_module(
        REPO / "ATS/models/ats_06_cear_g_ca.py"
    )

    A.DEVICE = args.device
    A.SEED = int(meta.get("seed", 0))
    A.G_DIM = int(meta["g_dim"])
    A.COUPLER = meta["coupler"]
    A.METRIC_RANK = int(meta["metric_rank"])
    A.ALPHA_MIN = float(meta["alpha_min"])
    A.ALPHA_MAX = float(meta["alpha_max"])
    A.TEXT_ENCODER = meta["text_encoder"]

    model = A.CEARHumanStateModel(
        int(meta["semantic_embed_dim"])
    ).to(args.device)

    ckpt = torch.load(
        atsdir / "best.pt",
        map_location="cpu",
        weights_only=False,
    )

    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    print("=" * 80)
    print("ATS08a — PersonaMem zero-training stress transfer")
    print("ATS06 best epoch:", meta.get("best_epoch"))
    print("device:", args.device)
    print("=" * 80)

    bench = pd.read_csv(
        root / "benchmark/text/benchmark.csv",
        dtype={"persona_id": str},
    )

    people = (
        bench[["persona_id", "raw_persona_file"]]
        .dropna()
        .drop_duplicates("persona_id")
    )

    rng = np.random.default_rng(args.seed)

    cases = []

    for i in rng.permutation(len(people)):
        row = people.iloc[int(i)]

        pid = str(row["persona_id"])
        remote = clean(row["raw_persona_file"])

        raw = json.loads(
            find_raw(root, remote).read_text(encoding="utf-8")
        )

        pairs = usable_pairs(A, raw)

        if not pairs:
            continue

        # exactly one native transition per persona
        pair = pairs[int(rng.integers(0, len(pairs)))]

        pair["persona_id"] = pid
        pair["raw_file"] = remote

        cases.append(pair)

    print("usable benchmark personas:", len(cases))

    if len(cases) < 20:
        raise RuntimeError("Too few usable benchmark personas.")

    items = []

    for c in cases:
        c["old_idx"] = len(items)
        items.append(c["old"])

        c["new_idx"] = len(items)
        items.append(c["new"])

    E = embed_items(
        A,
        items,
        meta["text_encoder"],
        args.device,
    )

    trajectory = []
    summary = []

    PRE = 6
    STEPS = 8
    RECOVERY = 4

    for n, c in enumerate(cases, 1):

        zo = E[c["old_idx"]]
        zn = E[c["new_idx"]]

        yo = c["old"]["choice"]
        yn = c["new"]["choice"]

        g = torch.zeros(A.G_DIM, device=args.device)
        err = A.initial_error_state(1, args.device)

        # establish pre-change state
        for _ in range(PRE):
            g, err, _ = advance(A, model, zo, yo, g, err)

        g0 = g.clone()
        e0 = clone_err(err)

        pnew0 = p_correct(model, zn, yn, g0)
        pold0 = p_correct(model, zo, yo, g0)

        # --------------------------------------------------
        # stable branch
        # --------------------------------------------------
        g = g0.clone()
        err = clone_err(e0)

        stable = {}

        for t in range(1, STEPS + 1):
            g, err, alpha = advance(
                A, model, zo, yo, g, err
            )

            stable[t] = l2(g, g0)

            trajectory.append({
                "persona_id": c["persona_id"],
                "condition": "stable_old",
                "step": t,
                "g_disp": stable[t],
                "g_cos_pre": cosine(g, g0),
                "p_new": p_correct(model, zn, yn, g),
                "p_old": p_correct(model, zo, yo, g),
                "alpha": alpha,
            })

        # --------------------------------------------------
        # sustained-change branch
        # --------------------------------------------------
        g = g0.clone()
        err = clone_err(e0)

        change = {}
        pnew = {}
        alpha_new = {}

        for t in range(1, STEPS + 1):

            g, err, alpha = advance(
                A, model, zn, yn, g, err
            )

            change[t] = l2(g, g0)
            pnew[t] = p_correct(model, zn, yn, g)
            alpha_new[t] = alpha

            trajectory.append({
                "persona_id": c["persona_id"],
                "condition": "sustained_new",
                "step": t,
                "g_disp": change[t],
                "g_cos_pre": cosine(g, g0),
                "p_new": pnew[t],
                "p_old": p_correct(model, zo, yo, g),
                "alpha": alpha,
            })

        g_change = g.clone()

        # --------------------------------------------------
        # one-shot contradiction -> recovery
        # --------------------------------------------------
        g = g0.clone()
        err = clone_err(e0)

        g, err, alpha = advance(
            A, model, zn, yn, g, err
        )

        shock = l2(g, g0)

        trajectory.append({
            "persona_id": c["persona_id"],
            "condition": "shock_recovery",
            "step": 0,
            "g_disp": shock,
            "g_cos_pre": cosine(g, g0),
            "p_new": p_correct(model, zn, yn, g),
            "p_old": p_correct(model, zo, yo, g),
            "alpha": alpha,
        })

        for t in range(1, RECOVERY + 1):

            g, err, alpha = advance(
                A, model, zo, yo, g, err
            )

            trajectory.append({
                "persona_id": c["persona_id"],
                "condition": "shock_recovery",
                "step": t,
                "g_disp": l2(g, g0),
                "g_cos_pre": cosine(g, g0),
                "p_new": p_correct(model, zn, yn, g),
                "p_old": p_correct(model, zo, yo, g),
                "alpha": alpha,
            })

        recovery_disp = l2(g, g0)

        summary.append({
            "persona_id": c["persona_id"],
            "old_pref": c["old_pref"],
            "new_pref": c["new_pref"],

            "p_new_pre": pnew0,
            "p_old_pre": pold0,

            "stable_final_disp": stable[8],

            "change_1_disp": change[1],
            "change_2_disp": change[2],
            "change_4_disp": change[4],
            "change_8_disp": change[8],

            "excess_change_final":
                change[8] - stable[8],

            "sustained_minus_oneoff":
                change[8] - change[1],

            "p_new_1": pnew[1],
            "p_new_2": pnew[2],
            "p_new_4": pnew[4],
            "p_new_8": pnew[8],

            "delta_p_new_final":
                pnew[8] - pnew0,

            "p_old_final":
                p_correct(model, zo, yo, g_change),

            "delta_p_old_final":
                p_correct(model, zo, yo, g_change) - pold0,

            "alpha_new_1": alpha_new[1],
            "alpha_new_8": alpha_new[8],

            "shock_disp": shock,
            "recovery_disp": recovery_disp,

            "recovery_fraction":
                1.0 - recovery_disp / shock
                if shock > EPS else np.nan,
        })

        if n % 25 == 0 or n == len(cases):
            print(f"processed {n}/{len(cases)}")

    S = pd.DataFrame(summary)
    T = pd.DataFrame(trajectory)

    S.to_csv(
        result / "pair_level_summary.csv",
        index=False,
    )

    T.to_csv(
        result / "trajectories.csv",
        index=False,
    )

    # ------------------------------------------------------
    # dose response
    # ------------------------------------------------------
    stable = T[T.condition == "stable_old"][
        ["persona_id", "step", "g_disp"]
    ]

    change = T[T.condition == "sustained_new"].copy()

    D = change.merge(
        stable,
        on=["persona_id", "step"],
        suffixes=("_change", "_stable"),
    )

    D["excess_g_disp"] = (
        D.g_disp_change - D.g_disp_stable
    )

    p0 = dict(zip(S.persona_id, S.p_new_pre))

    D["p_new_gain"] = [
        r.p_new - p0[r.persona_id]
        for _, r in D.iterrows()
    ]

    dose_rows = []

    for step, g in D.groupby("step"):

        for metric in [
            "g_disp_change",
            "g_disp_stable",
            "excess_g_disp",
            "p_new_gain",
            "alpha",
        ]:

            b = bootstrap(
                g[metric],
                seed=args.seed + 100 * int(step),
            )

            dose_rows.append({
                "step": step,
                "metric": metric,
                **b,
            })

    pd.DataFrame(dose_rows).to_csv(
        result / "dose_summary.csv",
        index=False,
    )

    # ------------------------------------------------------
    # overall summary
    # ------------------------------------------------------
    metrics = {}

    for i, metric in enumerate([
        "excess_change_final",
        "sustained_minus_oneoff",
        "delta_p_new_final",
        "delta_p_old_final",
        "recovery_fraction",
    ]):

        metrics[metric] = bootstrap(
            S[metric],
            seed=args.seed + i + 1,
        )

    fractions = {
        "change_final_gt_stable":
            float((S.change_8_disp > S.stable_final_disp).mean()),

        "change_final_gt_oneoff":
            float((S.change_8_disp > S.change_1_disp).mean()),

        "p_new_final_increases":
            float((S.delta_p_new_final > 0).mean()),

        "p_old_final_decreases":
            float((S.delta_p_old_final < 0).mean()),

        "positive_recovery_fraction":
            float((S.recovery_fraction > 0).mean()),
    }

    out = {
        "probe_type":
            "zero-training CA-to-PersonaMem cross-dataset transfer",

        "n_personas": len(S),

        "ats06_best_epoch":
            meta.get("best_epoch"),

        "seed": args.seed,

        "protocol": {
            "pre_old": PRE,
            "stress_steps": STEPS,
            "recovery_old": RECOVERY,
        },

        "metrics": metrics,
        "fractions": fractions,

        "interpretation_limit":
            "Weak transfer does not falsify ATS; "
            "the checkpoint was trained on Community Alignment.",
    }

    (
        result / "stress_summary.json"
    ).write_text(
        json.dumps(out, indent=2)
    )

    print()
    print("=" * 80)
    print("ATS08a STRESS SUMMARY")
    print("=" * 80)

    for k, v in metrics.items():
        print(
            f"{k:27s} "
            f"{v['mean']:+.5f} "
            f"95%CI[{v['ci_low']:+.5f},"
            f"{v['ci_high']:+.5f}]"
        )

    print()
    print("Fractions:")
    for k, v in fractions.items():
        print(f"  {k:30s} {v:.3f}")

    print()
    print("Desired signature:")
    print("  excess_change_final    > 0")
    print("  sustained_minus_oneoff > 0")
    print("  delta_p_new_final      > 0")
    print("  recovery_fraction      > 0")

    print()
    print("Wrote:", result)


if __name__ == "__main__":
    main()
