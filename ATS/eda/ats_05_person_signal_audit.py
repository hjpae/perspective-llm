"""
ATS 05 — Person-specific predictive-signal audit on Community Alignment
=======================================================================

Question
--------
Before training a neural ATS/g, does a person's *distributed interaction history*
contain information that improves held-out preference prediction beyond a strong
item-level baseline?

This is deliberately NOT yet an ATS model. It is a falsification / feasibility
test for whether there is person-specific signal worth compressing into g.

Design
------
* Community Alignment Wave 1 only.
* First-turn, pregenerated-prompt comparisons only.
  These provide the cleanest repeated exact comparisons across many people.
* Train / validation / test split is by USER (no user leakage).
* Each validation/test user's eligible comparisons are randomly split into:
      support evidence set  +  held-out query set
  There is NO assumed chronology; support is an unordered evidence set.
* Exact comparison identity is ORDER-INVARIANT:
      normalized prompt + sorted candidate texts
  The observed selected candidate is remapped to this canonical ordering.
  Therefore models cannot exploit the literal A/B/C/D label as the target.
* Only items with enough TRAIN-user ratings are evaluated.

Baselines
---------
B0  Global prior.
B1  Exact-comparison prior from TRAIN users.
B2  Exact-comparison + language prior from TRAIN users.
B3  B2 + target user's historical A/B/C/D position tendency
    (nuisance/person-style control).
B4  Personalized collaborative kNN:
    infer which TRAIN users agree with the target on the target's SUPPORT items,
    then use those neighbors to reweight B2 on held-out query items.
B5  Same B4, but the target's support history is replaced by another TEST user's
    support history from the SAME LANGUAGE. This is the wrong-person control.

Interpretation
--------------
The key signal is:
    B4(real support) > B2(item+language)
and, more strongly:
    B4(real support) > B5(same-language wrong-person support)

If neither is true, a learned g has little empirical justification on this
subset. If they are true, the next question is whether a compact learned g can
capture the gain more efficiently/generalizably than explicit support history.

This script streams the public Hugging Face dataset and saves only compact
hashed interaction records / aggregate outputs. Raw corpus text is NOT saved.

Expected repo location
----------------------
perspective-llm/ATS/evaluation/ats_05_person_signal_audit.py

Dependencies
------------
pip install datasets huggingface_hub pandas numpy
"""

# %% Imports
from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from datasets import load_dataset
from huggingface_hub import get_token


# %% Paths / configuration
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]
RESULT_ROOT = REPO_ROOT / "results" / "ATS" / "person_signal_audit"

HF_DATASET = "facebook/community-alignment-dataset"
HF_TOKEN = os.environ.get("HF_TOKEN") or get_token()

SEED = 20260922

# User-level population split within Wave 1.
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
# remainder is test

# Clean repeated-comparison benchmark.
USE_WAVE = "1"
REQUIRE_PREGENERATED_FIRST_PROMPT = True
MIN_TRAIN_ITEM_RATERS = 8

# Each unseen user's unordered evidence/query split.
SUPPORT_FRAC = 0.67
MIN_SUPPORT = 6
MIN_QUERY = 3

# Neighbor must share at least this many support comparisons with target.
MIN_NEIGHBOR_OVERLAP = 2

# Hyperparameters tuned ONLY on validation users.
K_GRID = [10, 25, 50, 100]
SHRINK_GRID = [2.0, 5.0, 10.0]
ETA_GRID = [0.25, 0.50, 0.75]

# Position-profile nuisance baseline mixing weight.
POSITION_ETA_GRID = [0.10, 0.25, 0.50, 0.75]

# Wrong-person control.
N_SHUFFLES = 25

# User-cluster bootstrap for uncertainty.
N_BOOTSTRAP = 3000
BOOTSTRAP_SEED = 20260923

PROGRESS_EVERY = 5000
EPS = 1e-12
N_CHOICES = 4

POSITIONS = ("a", "b", "c", "d")


# %% Text / hashing utilities
def clean_text(x: Any) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x).strip()


def normalize_text(x: Any) -> str:
    s = clean_text(x).lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s


def stable_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


def normalize_choice(x: Any) -> str:
    s = clean_text(x).lower().strip()
    mapping = {
        "a": "a", "b": "b", "c": "c", "d": "d",
        "response_a": "a", "response_b": "b",
        "response_c": "c", "response_d": "d",
        "option_a": "a", "option_b": "b",
        "option_c": "c", "option_d": "d",
        "response a": "a", "response b": "b",
        "response c": "c", "response d": "d",
        "option a": "a", "option b": "b",
        "option c": "c", "option d": "d",
    }
    if s in mapping:
        return mapping[s]
    m = re.fullmatch(r"\s*([abcd])[\.\):]?\s*", s)
    return m.group(1) if m else ""


def as_bool(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    s = clean_text(x).lower()
    return s in {"1", "true", "t", "yes", "y"}


def canonicalize_item(row: Dict[str, Any]):
    """Return order-invariant item identity and canonical selected candidate.

    Candidate texts are normalized then sorted by content hash. This means the
    target label is candidate identity in a canonical ordering, not A/B/C/D.
    """
    prompt = normalize_text(row.get("first_turn_prompt"))
    raw_choice = normalize_choice(row.get("first_turn_preferred_response"))

    if not prompt or not raw_choice:
        return None

    candidates = {
        p: normalize_text(row.get(f"first_turn_response_{p}"))
        for p in POSITIONS
    }
    if not all(candidates.values()):
        return None

    # Duplicate candidates make semantic remapping ambiguous; skip them.
    candidate_hash_by_pos = {
        p: stable_hash(candidates[p]) for p in POSITIONS
    }
    if len(set(candidate_hash_by_pos.values())) != N_CHOICES:
        return "DUPLICATE_CANDIDATES"

    sorted_hashes = sorted(candidate_hash_by_pos.values())
    hash_to_canon = {h: i for i, h in enumerate(sorted_hashes)}

    chosen_hash = candidate_hash_by_pos[raw_choice]
    canonical_choice = hash_to_canon[chosen_hash]

    # Exact comparison identity ignores original candidate ordering.
    item_id = stable_hash(
        stable_hash(prompt) + "|" + "|".join(sorted_hashes)
    )

    # For the nuisance position baseline, keep how original A/B/C/D positions
    # map into canonical candidate identities for THIS query.
    pos_to_canon = tuple(
        hash_to_canon[candidate_hash_by_pos[p]] for p in POSITIONS
    )

    return {
        "item_id": item_id,
        "choice": int(canonical_choice),
        "raw_position": raw_choice,
        "pos_to_canon": pos_to_canon,
    }


# %% Compact dataset extraction
def stream_compact_wave1() -> pd.DataFrame:
    print("Streaming Community Alignment from Hugging Face...")
    print("Raw texts are hashed in memory and are NOT persisted.\n")

    ds = load_dataset(
        HF_DATASET,
        split="train",
        streaming=True,
        token=HF_TOKEN,
    )

    compact = []
    duplicate_candidate_rows = 0
    skipped = Counter()

    for i, row0 in enumerate(ds, 1):
        row = dict(row0)

        if clean_text(row.get("wave")) != USE_WAVE:
            skipped["other_wave"] += 1
            continue

        if REQUIRE_PREGENERATED_FIRST_PROMPT and not as_bool(
            row.get("is_pregenerated_first_prompt")
        ):
            skipped["user_initiated_first_prompt"] += 1
            continue

        uid = clean_text(row.get("annotator_id"))
        lang = clean_text(row.get("assigned_lang"))
        if not uid or not lang:
            skipped["missing_user_or_language"] += 1
            continue

        item = canonicalize_item(row)
        if item is None:
            skipped["missing_prompt_candidate_or_choice"] += 1
            continue
        if item == "DUPLICATE_CANDIDATES":
            duplicate_candidate_rows += 1
            continue

        compact.append({
            "user_id": uid,
            "language": lang,
            "item_id": item["item_id"],
            "choice": item["choice"],
            "raw_position": item["raw_position"],
            "pos_a_canon": item["pos_to_canon"][0],
            "pos_b_canon": item["pos_to_canon"][1],
            "pos_c_canon": item["pos_to_canon"][2],
            "pos_d_canon": item["pos_to_canon"][3],
        })

        if i % PROGRESS_EVERY == 0:
            print(
                f"  streamed {i:,} raw rows | "
                f"kept compact interactions={len(compact):,}"
            )

    df = pd.DataFrame(compact)

    print("\n[stream summary]")
    print("compact interactions:", f"{len(df):,}")
    print("users:", f"{df['user_id'].nunique():,}")
    print("items:", f"{df['item_id'].nunique():,}")
    print("duplicate-candidate rows skipped:", f"{duplicate_candidate_rows:,}")
    print("other skips:", dict(skipped))

    return df


# %% De-duplicate repeated exposure of same exact item by same user
def dedupe_user_item(df: pd.DataFrame):
    rows = []
    same_choice_collapsed = 0
    conflicting_dropped = 0

    for (uid, item), g in df.groupby(["user_id", "item_id"], sort=False):
        choices = g["choice"].unique()
        if len(choices) > 1:
            conflicting_dropped += 1
            continue
        if len(g) > 1:
            same_choice_collapsed += len(g) - 1
        rows.append(g.iloc[0])

    out = pd.DataFrame(rows).reset_index(drop=True)
    return out, same_choice_collapsed, conflicting_dropped


# %% User split
def modal_language_table(df: pd.DataFrame):
    langs = (
        df.groupby("user_id")["language"]
        .agg(lambda x: tuple(sorted(set(x))))
        .reset_index()
    )
    inconsistent = set(
        langs.loc[langs["language"].map(len) > 1, "user_id"].astype(str)
    )
    return inconsistent


def stratified_user_split(df: pd.DataFrame):
    """Deterministic user-disjoint split stratified by modal/unique language."""
    rng = np.random.default_rng(SEED)

    user_lang = (
        df.groupby("user_id")["language"]
        .first()
        .reset_index()
    )

    assignments = []
    for lang, g in user_lang.groupby("language"):
        users = g["user_id"].astype(str).to_numpy()
        rng.shuffle(users)

        n = len(users)
        n_train = int(round(TRAIN_FRAC * n))
        n_val = int(round(VAL_FRAC * n))
        n_train = min(n_train, n)
        n_val = min(n_val, n - n_train)

        for u in users[:n_train]:
            assignments.append((u, lang, "train"))
        for u in users[n_train:n_train + n_val]:
            assignments.append((u, lang, "val"))
        for u in users[n_train + n_val:]:
            assignments.append((u, lang, "test"))

    return pd.DataFrame(
        assignments, columns=["user_id", "language", "split"]
    )


# %% Probability helpers
def smooth_probs(counts, alpha=1.0):
    x = np.asarray(counts, dtype=float)
    return (x + alpha) / (x.sum() + alpha * len(x))


def normalize_probs(x):
    x = np.asarray(x, dtype=float)
    x = np.maximum(x, 0.0)
    s = x.sum()
    if s <= 0:
        return np.ones(N_CHOICES) / N_CHOICES
    return x / s


# %% Train-population statistics
@dataclass
class TrainIndex:
    global_probs: np.ndarray
    item_probs: Dict[str, np.ndarray]
    lang_item_probs: Dict[Tuple[str, str], np.ndarray]
    by_item: Dict[str, Dict[str, int]]
    by_user: Dict[str, Dict[str, int]]
    item_train_n: Counter


def build_train_index(train_df: pd.DataFrame) -> TrainIndex:
    global_counts = np.bincount(
        train_df["choice"].to_numpy(dtype=int), minlength=N_CHOICES
    )
    global_probs = smooth_probs(global_counts, alpha=1.0)

    by_item = defaultdict(dict)
    by_user = defaultdict(dict)
    item_counts = defaultdict(lambda: np.zeros(N_CHOICES, dtype=int))
    lang_item_counts = defaultdict(lambda: np.zeros(N_CHOICES, dtype=int))
    item_train_n = Counter()

    for r in train_df.itertuples(index=False):
        c = int(r.choice)
        by_item[r.item_id][r.user_id] = c
        by_user[r.user_id][r.item_id] = c
        item_counts[r.item_id][c] += 1
        lang_item_counts[(r.language, r.item_id)][c] += 1
        item_train_n[r.item_id] += 1

    item_probs = {
        item: smooth_probs(counts, alpha=1.0)
        for item, counts in item_counts.items()
    }

    # Language-item estimate is shrunk toward overall item probability.
    lang_item_probs = {}
    tau = 4.0
    for key, counts in lang_item_counts.items():
        lang, item = key
        base = item_probs[item]
        lang_item_probs[key] = normalize_probs(counts + tau * base)

    return TrainIndex(
        global_probs=global_probs,
        item_probs=item_probs,
        lang_item_probs=lang_item_probs,
        by_item=dict(by_item),
        by_user=dict(by_user),
        item_train_n=item_train_n,
    )


# %% Support/query creation for unseen users
@dataclass
class TargetUser:
    user_id: str
    language: str
    support: pd.DataFrame
    query: pd.DataFrame


def deterministic_user_seed(uid: str) -> int:
    h = hashlib.sha1(f"{SEED}|{uid}".encode()).hexdigest()[:8]
    return int(h, 16)


def make_targets(
    df: pd.DataFrame,
    split_users: set,
    train_index: TrainIndex,
):
    eligible = df[
        df["user_id"].isin(split_users)
        & df["item_id"].map(
            lambda x: train_index.item_train_n.get(x, 0)
            >= MIN_TRAIN_ITEM_RATERS
        )
    ].copy()

    targets = []
    dropped = Counter()

    for uid, g in eligible.groupby("user_id", sort=False):
        g = g.drop_duplicates("item_id").reset_index(drop=True)
        n = len(g)

        if n < MIN_SUPPORT + MIN_QUERY:
            dropped["too_few_eligible_interactions"] += 1
            continue

        rng = np.random.default_rng(deterministic_user_seed(str(uid)))
        idx = np.arange(n)
        rng.shuffle(idx)

        n_support = int(round(SUPPORT_FRAC * n))
        n_support = max(MIN_SUPPORT, n_support)
        n_support = min(n_support, n - MIN_QUERY)

        support = g.iloc[idx[:n_support]].copy()
        query = g.iloc[idx[n_support:]].copy()

        # Hard guarantee: no exact item is both support and query.
        assert set(support["item_id"]).isdisjoint(set(query["item_id"]))

        targets.append(
            TargetUser(
                user_id=str(uid),
                language=str(g["language"].iloc[0]),
                support=support,
                query=query,
            )
        )

    return targets, eligible, dropped


# %% Baselines
def base_probs_for_query(
    item_id: str,
    language: str,
    train_index: TrainIndex,
    mode: str,
):
    if mode == "global":
        return train_index.global_probs

    if mode == "item":
        return train_index.item_probs.get(
            item_id, train_index.global_probs
        )

    if mode == "lang_item":
        return train_index.lang_item_probs.get(
            (language, item_id),
            train_index.item_probs.get(item_id, train_index.global_probs),
        )

    raise ValueError(mode)


def support_position_probs(support: pd.DataFrame):
    counts = np.ones(4, dtype=float)  # mild Dirichlet smoothing
    pos_index = {"a": 0, "b": 1, "c": 2, "d": 3}
    for p in support["raw_position"]:
        if p in pos_index:
            counts[pos_index[p]] += 1.0
    return counts / counts.sum()


def mapped_position_probs(query_row, position_probs):
    pos_to_canon = [
        int(query_row.pos_a_canon),
        int(query_row.pos_b_canon),
        int(query_row.pos_c_canon),
        int(query_row.pos_d_canon),
    ]
    out = np.zeros(N_CHOICES, dtype=float)
    for raw_pos_idx, canon_idx in enumerate(pos_to_canon):
        out[canon_idx] += position_probs[raw_pos_idx]
    return normalize_probs(out)


def compute_neighbor_stats(
    support: pd.DataFrame,
    train_index: TrainIndex,
):
    """Chance-corrected agreement with each TRAIN user on support items.

    For support item j with target choice c:
        expected agreement under item prior = p_j(c)
        contribution if neighbor agrees    = 1 - p_j(c)
        contribution otherwise             = -p_j(c)

    Sum has expectation ~0 for a random train user drawn from the item's
    population, so globally easy items contribute less than discriminative ones.
    """
    residual = defaultdict(float)
    overlap = Counter()

    for r in support.itertuples(index=False):
        item = r.item_id
        target_choice = int(r.choice)
        p = train_index.item_probs[item][target_choice]

        for train_user, train_choice in train_index.by_item[item].items():
            overlap[train_user] += 1
            residual[train_user] += (
                (1.0 - p) if train_choice == target_choice else (-p)
            )

    return residual, overlap


def personalized_knn_probs(
    item_id: str,
    language: str,
    neighbor_stats,
    train_index: TrainIndex,
    k: int,
    shrink: float,
    eta: float,
):
    base = base_probs_for_query(
        item_id, language, train_index, mode="lang_item"
    )

    residual, overlap = neighbor_stats
    raters = train_index.by_item.get(item_id, {})

    scored = []
    for train_user, choice in raters.items():
        ov = overlap.get(train_user, 0)
        if ov < MIN_NEIGHBOR_OVERLAP:
            continue
        sim = residual.get(train_user, 0.0) / (ov + shrink)
        if sim > 0:
            scored.append((sim, int(choice)))

    if not scored:
        return base.copy(), 0

    scored.sort(key=lambda z: z[0], reverse=True)
    if k is not None:
        scored = scored[:k]

    vote = np.zeros(N_CHOICES, dtype=float)
    for weight, choice in scored:
        vote[choice] += weight

    if vote.sum() <= 0:
        return base.copy(), 0

    neighbor_dist = vote / vote.sum()
    pred = (1.0 - eta) * base + eta * neighbor_dist
    return normalize_probs(pred), len(scored)


# %% Metrics
def row_metrics(true_choice: int, probs: np.ndarray):
    pred = int(np.argmax(probs))
    p_true = float(np.clip(probs[true_choice], EPS, 1.0))
    onehot = np.zeros(N_CHOICES, dtype=float)
    onehot[true_choice] = 1.0
    return {
        "correct": int(pred == true_choice),
        "nll": -math.log(p_true),
        "brier": float(np.sum((probs - onehot) ** 2)),
        "p_true": p_true,
        "pred": pred,
    }


def summarize_predictions(pred_df: pd.DataFrame):
    out = []
    for model, g in pred_df.groupby("model"):
        out.append({
            "model": model,
            "n_query": len(g),
            "n_users": g["user_id"].nunique(),
            "accuracy": g["correct"].mean(),
            "nll": g["nll"].mean(),
            "brier": g["brier"].mean(),
            "mean_p_true": g["p_true"].mean(),
            "mean_neighbors_used": (
                g["neighbors_used"].mean()
                if "neighbors_used" in g else np.nan
            ),
        })
    return pd.DataFrame(out).sort_values("nll")


# %% Evaluate target users
def evaluate_targets(
    targets: List[TargetUser],
    train_index: TrainIndex,
    knn_params=None,
    position_eta=None,
    support_override: Dict[str, pd.DataFrame] | None = None,
    models=("global", "item", "lang_item", "position", "knn"),
):
    rows = []

    for target in targets:
        support = (
            support_override[target.user_id]
            if support_override is not None
            else target.support
        )

        pos_probs = support_position_probs(support)
        neighbor_stats = (
            compute_neighbor_stats(support, train_index)
            if "knn" in models else None
        )

        for q in target.query.itertuples(index=False):
            true = int(q.choice)

            if "global" in models:
                probs = base_probs_for_query(
                    q.item_id, target.language, train_index, "global"
                )
                m = row_metrics(true, probs)
                rows.append({
                    "user_id": target.user_id,
                    "language": target.language,
                    "item_id": q.item_id,
                    "model": "B0_global",
                    "neighbors_used": 0,
                    **m,
                })

            if "item" in models:
                probs = base_probs_for_query(
                    q.item_id, target.language, train_index, "item"
                )
                m = row_metrics(true, probs)
                rows.append({
                    "user_id": target.user_id,
                    "language": target.language,
                    "item_id": q.item_id,
                    "model": "B1_item",
                    "neighbors_used": 0,
                    **m,
                })

            if "lang_item" in models:
                probs = base_probs_for_query(
                    q.item_id, target.language, train_index, "lang_item"
                )
                m = row_metrics(true, probs)
                rows.append({
                    "user_id": target.user_id,
                    "language": target.language,
                    "item_id": q.item_id,
                    "model": "B2_lang_item",
                    "neighbors_used": 0,
                    **m,
                })

            if "position" in models:
                base = base_probs_for_query(
                    q.item_id, target.language, train_index, "lang_item"
                )
                qpos = mapped_position_probs(q, pos_probs)
                eta = float(position_eta)
                probs = normalize_probs((1.0 - eta) * base + eta * qpos)
                m = row_metrics(true, probs)
                rows.append({
                    "user_id": target.user_id,
                    "language": target.language,
                    "item_id": q.item_id,
                    "model": "B3_position_profile",
                    "neighbors_used": 0,
                    **m,
                })

            if "knn" in models:
                probs, n_nb = personalized_knn_probs(
                    item_id=q.item_id,
                    language=target.language,
                    neighbor_stats=neighbor_stats,
                    train_index=train_index,
                    k=int(knn_params["k"]),
                    shrink=float(knn_params["shrink"]),
                    eta=float(knn_params["eta"]),
                )
                m = row_metrics(true, probs)
                rows.append({
                    "user_id": target.user_id,
                    "language": target.language,
                    "item_id": q.item_id,
                    "model": "B4_person_knn",
                    "neighbors_used": n_nb,
                    **m,
                })

    return pd.DataFrame(rows)


# %% Validation tuning
def tune_position_eta(val_targets, train_index):
    rows = []
    for eta in POSITION_ETA_GRID:
        pred = evaluate_targets(
            val_targets,
            train_index,
            position_eta=eta,
            models=("position",),
        )
        rows.append({
            "eta": eta,
            "nll": pred["nll"].mean(),
            "accuracy": pred["correct"].mean(),
        })
    tab = pd.DataFrame(rows).sort_values(["nll", "accuracy"], ascending=[True, False])
    best = float(tab.iloc[0]["eta"])
    return best, tab


def tune_knn(val_targets, train_index):
    rows = []

    # Compute one configuration at a time; corpus is small enough that this is
    # transparent and easy to audit.
    for k in K_GRID:
        for shrink in SHRINK_GRID:
            for eta in ETA_GRID:
                params = {"k": k, "shrink": shrink, "eta": eta}
                pred = evaluate_targets(
                    val_targets,
                    train_index,
                    knn_params=params,
                    models=("knn",),
                )
                rows.append({
                    **params,
                    "n_query": len(pred),
                    "nll": pred["nll"].mean(),
                    "accuracy": pred["correct"].mean(),
                    "mean_neighbors_used": pred["neighbors_used"].mean(),
                })
                print(
                    "  tune",
                    params,
                    f"NLL={rows[-1]['nll']:.4f}",
                    f"acc={rows[-1]['accuracy']:.3f}",
                )

    tab = pd.DataFrame(rows).sort_values(
        ["nll", "accuracy"], ascending=[True, False]
    )
    best_row = tab.iloc[0]
    best = {
        "k": int(best_row["k"]),
        "shrink": float(best_row["shrink"]),
        "eta": float(best_row["eta"]),
    }
    return best, tab


# %% Same-language wrong-person controls
def same_language_donor_map(targets, rng):
    by_lang = defaultdict(list)
    for t in targets:
        by_lang[t.language].append(t.user_id)

    donor = {}
    for lang, users0 in by_lang.items():
        users = list(users0)
        if len(users) <= 1:
            continue

        # Shuffle then cyclically shift until self-matches are removed.
        perm = users.copy()
        rng.shuffle(perm)

        # Rotate by one; because source and perm are differently ordered there
        # can still be rare self-matches. Repair those by swapping donors.
        donors = perm[1:] + perm[:1]
        for _ in range(len(users) * 2):
            bad = [i for i, u in enumerate(users) if donors[i] == u]
            if not bad:
                break
            i = bad[0]
            j = (i + 1) % len(users)
            donors[i], donors[j] = donors[j], donors[i]

        for u, d in zip(users, donors):
            if u != d:
                donor[u] = d

    return donor


def evaluate_shuffled_support(
    test_targets,
    train_index,
    best_knn,
):
    target_by_uid = {t.user_id: t for t in test_targets}
    rng = np.random.default_rng(SEED + 991)

    all_summary = []
    user_metric_accum = defaultdict(list)

    for rep in range(N_SHUFFLES):
        donor_map = same_language_donor_map(test_targets, rng)
        override = {}

        kept_targets = []
        for t in test_targets:
            donor_uid = donor_map.get(t.user_id)
            if donor_uid is None:
                continue
            override[t.user_id] = target_by_uid[donor_uid].support
            kept_targets.append(t)

        pred = evaluate_targets(
            kept_targets,
            train_index,
            knn_params=best_knn,
            support_override=override,
            models=("knn",),
        )
        pred["model"] = "B5_wrong_person_knn"

        all_summary.append({
            "shuffle": rep,
            "n_query": len(pred),
            "accuracy": pred["correct"].mean(),
            "nll": pred["nll"].mean(),
            "brier": pred["brier"].mean(),
            "mean_neighbors_used": pred["neighbors_used"].mean(),
        })

        per_user = pred.groupby("user_id").agg(
            accuracy=("correct", "mean"),
            nll=("nll", "mean"),
            brier=("brier", "mean"),
        )
        for uid, r in per_user.iterrows():
            user_metric_accum[uid].append(
                (float(r["accuracy"]), float(r["nll"]), float(r["brier"]))
            )

    # Average wrong-person performance per target user across shuffles.
    user_rows = []
    for uid, vals in user_metric_accum.items():
        arr = np.asarray(vals)
        user_rows.append({
            "user_id": uid,
            "wrong_accuracy": arr[:, 0].mean(),
            "wrong_nll": arr[:, 1].mean(),
            "wrong_brier": arr[:, 2].mean(),
        })

    return pd.DataFrame(all_summary), pd.DataFrame(user_rows)


# %% Cluster bootstrap over users
def clustered_bootstrap_delta(
    pred_df: pd.DataFrame,
    model_a: str,
    model_b: str,
    metric: str,
    n_boot=N_BOOTSTRAP,
):
    """Delta = model_a - model_b, aggregating metric within user first.

    Raw prediction rows store accuracy as the binary `correct` column, whereas
    summary tables call its mean `accuracy`. Map the summary metric name back
    to the raw column here.
    """
    metric_col = {
        "accuracy": "correct",
        "nll": "nll",
        "brier": "brier",
    }[metric]

    a = (
        pred_df[pred_df["model"] == model_a]
        .groupby("user_id")[metric_col]
        .mean()
    )
    b = (
        pred_df[pred_df["model"] == model_b]
        .groupby("user_id")[metric_col]
        .mean()
    )
    common = a.index.intersection(b.index)
    diffs = (a.loc[common] - b.loc[common]).to_numpy(dtype=float)

    if len(diffs) == 0:
        return {
            "model_a": model_a,
            "model_b": model_b,
            "metric": metric,
            "n_users": 0,
            "delta": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
        }

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    boots = np.empty(n_boot, dtype=float)
    n = len(diffs)
    for i in range(n_boot):
        sample = rng.integers(0, n, size=n)
        boots[i] = diffs[sample].mean()

    return {
        "model_a": model_a,
        "model_b": model_b,
        "metric": metric,
        "n_users": n,
        "delta": float(diffs.mean()),
        "ci_low": float(np.quantile(boots, 0.025)),
        "ci_high": float(np.quantile(boots, 0.975)),
    }


def bootstrap_real_vs_wrong(real_pred, wrong_user_df):
    real_user = (
        real_pred[real_pred["model"] == "B4_person_knn"]
        .groupby("user_id")
        .agg(
            real_accuracy=("correct", "mean"),
            real_nll=("nll", "mean"),
            real_brier=("brier", "mean"),
        )
        .reset_index()
    )
    m = real_user.merge(wrong_user_df, on="user_id", how="inner")

    rows = []
    rng = np.random.default_rng(BOOTSTRAP_SEED + 10)

    for metric in ("accuracy", "nll", "brier"):
        diffs = (
            m[f"real_{metric}"].to_numpy()
            - m[f"wrong_{metric}"].to_numpy()
        )
        boots = np.empty(N_BOOTSTRAP)
        n = len(diffs)
        for i in range(N_BOOTSTRAP):
            idx = rng.integers(0, n, size=n)
            boots[i] = diffs[idx].mean()

        rows.append({
            "model_a": "B4_person_knn",
            "model_b": "B5_wrong_person_knn",
            "metric": metric,
            "n_users": n,
            "delta": float(diffs.mean()),
            "ci_low": float(np.quantile(boots, 0.025)),
            "ci_high": float(np.quantile(boots, 0.975)),
        })

    return rows


# %% Main
def main():
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)

    print("REPO_ROOT:   ", REPO_ROOT)
    print("RESULT_ROOT: ", RESULT_ROOT)
    print("HF auth:     ", "yes" if HF_TOKEN else "no (public streaming)")
    print()

    # 1) Stream compact Wave-1 controlled comparisons.
    df = stream_compact_wave1()

    # 2) Remove same-user repeated exposure ambiguity.
    df, collapsed, conflicting = dedupe_user_item(df)
    print("\n[dedupe]")
    print("same-choice duplicate rows collapsed:", f"{collapsed:,}")
    print("conflicting repeated user-item pairs dropped:", f"{conflicting:,}")
    print("remaining interactions:", f"{len(df):,}")

    # 3) Exclude rare users with inconsistent assigned language.
    inconsistent_users = modal_language_table(df)
    if inconsistent_users:
        print(
            "dropping users with >1 assigned language:",
            len(inconsistent_users),
        )
        df = df[~df["user_id"].isin(inconsistent_users)].copy()

    # 4) USER-disjoint population split.
    split_df = stratified_user_split(df)
    split_df.to_csv(RESULT_ROOT / "user_splits.csv", index=False)

    split_map = dict(zip(split_df["user_id"], split_df["split"]))
    df["split"] = df["user_id"].map(split_map)

    print("\n[user split]")
    print(split_df["split"].value_counts())
    print("\n[user split × language]")
    print(pd.crosstab(split_df["language"], split_df["split"]))

    # 5) Build population baselines ONLY from TRAIN users.
    train_df_all = df[df["split"] == "train"].copy()
    train_index_all = build_train_index(train_df_all)

    eligible_items = {
        item
        for item, n in train_index_all.item_train_n.items()
        if n >= MIN_TRAIN_ITEM_RATERS
    }
    print(
        "\nitems with >=",
        MIN_TRAIN_ITEM_RATERS,
        "TRAIN-user raters:",
        f"{len(eligible_items):,}",
    )

    # Rebuild train index on benchmark-eligible exact comparisons only.
    train_df = train_df_all[
        train_df_all["item_id"].isin(eligible_items)
    ].copy()
    train_index = build_train_index(train_df)

    # 6) Build unseen validation/test target users.
    val_users = set(
        split_df.loc[split_df["split"] == "val", "user_id"].astype(str)
    )
    test_users = set(
        split_df.loc[split_df["split"] == "test", "user_id"].astype(str)
    )

    val_targets, val_eligible, val_drop = make_targets(
        df, val_users, train_index
    )
    test_targets, test_eligible, test_drop = make_targets(
        df, test_users, train_index
    )

    def target_summary(targets, name):
        n_support = [len(t.support) for t in targets]
        n_query = [len(t.query) for t in targets]
        print(f"\n[{name} target coverage]")
        print("users retained:", len(targets))
        if targets:
            print("median support:", float(np.median(n_support)))
            print("median query:  ", float(np.median(n_query)))
            print("total queries: ", int(sum(n_query)))

    target_summary(val_targets, "validation")
    print("validation drops:", dict(val_drop))
    target_summary(test_targets, "test")
    print("test drops:", dict(test_drop))

    # Save compact coverage, not raw text.
    coverage_rows = []
    for split_name, targets in [("val", val_targets), ("test", test_targets)]:
        for t in targets:
            coverage_rows.append({
                "split": split_name,
                "user_id": t.user_id,
                "language": t.language,
                "support_n": len(t.support),
                "query_n": len(t.query),
            })
    pd.DataFrame(coverage_rows).to_csv(
        RESULT_ROOT / "target_user_coverage.csv", index=False
    )

    # 7) Tune nuisance position-personalization control on validation.
    print("\nTuning historical position-profile weight on VALIDATION...")
    best_position_eta, pos_tuning = tune_position_eta(
        val_targets, train_index
    )
    pos_tuning.to_csv(
        RESULT_ROOT / "validation_position_tuning.csv", index=False
    )
    print("best position eta:", best_position_eta)

    # 8) Tune personalized kNN on validation only.
    print("\nTuning personalized kNN on VALIDATION...")
    best_knn, knn_tuning = tune_knn(val_targets, train_index)
    knn_tuning.to_csv(
        RESULT_ROOT / "validation_knn_tuning.csv", index=False
    )
    print("best kNN:", best_knn)

    # 9) Final untouched TEST evaluation.
    print("\nEvaluating untouched TEST users...")
    test_pred = evaluate_targets(
        test_targets,
        train_index,
        knn_params=best_knn,
        position_eta=best_position_eta,
        models=("global", "item", "lang_item", "position", "knn"),
    )
    test_pred.to_csv(
        RESULT_ROOT / "test_predictions.csv", index=False
    )

    test_summary = summarize_predictions(test_pred)
    test_summary.to_csv(
        RESULT_ROOT / "test_baseline_summary.csv", index=False
    )

    # 10) Same-language wrong-person support control.
    print("\nRunning same-language WRONG-PERSON support control...")
    shuffle_summary, wrong_user = evaluate_shuffled_support(
        test_targets,
        train_index,
        best_knn,
    )
    shuffle_summary.to_csv(
        RESULT_ROOT / "wrong_person_shuffle_summary.csv", index=False
    )
    wrong_user.to_csv(
        RESULT_ROOT / "wrong_person_per_user.csv", index=False
    )

    wrong_overall = {
        "model": "B5_wrong_person_knn",
        "n_query": int(shuffle_summary["n_query"].median()),
        "n_users": int(len(wrong_user)),
        "accuracy": float(shuffle_summary["accuracy"].mean()),
        "nll": float(shuffle_summary["nll"].mean()),
        "brier": float(shuffle_summary["brier"].mean()),
        "mean_p_true": np.nan,
        "mean_neighbors_used": float(
            shuffle_summary["mean_neighbors_used"].mean()
        ),
    }
    test_summary_with_wrong = pd.concat(
        [test_summary, pd.DataFrame([wrong_overall])],
        ignore_index=True,
    ).sort_values("nll")
    test_summary_with_wrong.to_csv(
        RESULT_ROOT / "test_baseline_summary_with_wrong_person.csv",
        index=False,
    )

    # 11) User-cluster bootstrap of key deltas.
    delta_rows = []
    for metric in ("accuracy", "nll", "brier"):
        delta_rows.append(
            clustered_bootstrap_delta(
                test_pred, "B4_person_knn", "B2_lang_item", metric
            )
        )
        delta_rows.append(
            clustered_bootstrap_delta(
                test_pred, "B4_person_knn", "B3_position_profile", metric
            )
        )

    delta_rows.extend(
        bootstrap_real_vs_wrong(test_pred, wrong_user)
    )

    delta_df = pd.DataFrame(delta_rows)
    delta_df.to_csv(
        RESULT_ROOT / "bootstrap_deltas.csv", index=False
    )

    # 12) Save benchmark design facts / reproducibility.
    design = {
        "wave": USE_WAVE,
        "pregenerated_first_prompt_only": REQUIRE_PREGENERATED_FIRST_PROMPT,
        "candidate_order_invariant_item_identity": True,
        "train_fraction": TRAIN_FRAC,
        "validation_fraction": VAL_FRAC,
        "test_fraction": 1.0 - TRAIN_FRAC - VAL_FRAC,
        "min_train_item_raters": MIN_TRAIN_ITEM_RATERS,
        "support_fraction": SUPPORT_FRAC,
        "min_support": MIN_SUPPORT,
        "min_query": MIN_QUERY,
        "min_neighbor_overlap": MIN_NEIGHBOR_OVERLAP,
        "best_position_eta": best_position_eta,
        "best_knn": best_knn,
        "n_train_users": int(
            (split_df["split"] == "train").sum()
        ),
        "n_val_users_raw": int(
            (split_df["split"] == "val").sum()
        ),
        "n_test_users_raw": int(
            (split_df["split"] == "test").sum()
        ),
        "n_val_users_retained": len(val_targets),
        "n_test_users_retained": len(test_targets),
        "eligible_exact_comparisons": len(eligible_items),
        "same_choice_duplicate_rows_collapsed": collapsed,
        "conflicting_repeated_user_item_pairs_dropped": conflicting,
        "multi_language_users_dropped": len(inconsistent_users),
        "wrong_person_shuffles": N_SHUFFLES,
        "bootstrap_replicates": N_BOOTSTRAP,
        "seed": SEED,
    }
    with open(
        RESULT_ROOT / "audit_design.json", "w", encoding="utf-8"
    ) as f:
        json.dump(design, f, indent=2)

    # 13) Console report.
    print("\n" + "=" * 78)
    print("PERSON-SPECIFIC PREDICTIVE SIGNAL AUDIT — TEST")
    print("=" * 78)
    print(
        test_summary_with_wrong[
            ["model", "n_query", "n_users", "accuracy", "nll", "brier",
             "mean_neighbors_used"]
        ].to_string(index=False)
    )

    print("\n[key user-cluster bootstrap deltas: model_a - model_b]")
    print(delta_df.to_string(index=False))

    print("\nInterpretation guide:")
    print("  Accuracy: positive B4-B2 is good for person-specific signal.")
    print("  NLL/Brier: negative B4-B2 is good.")
    print("  B4 vs B3 asks whether gain exceeds simple personal position bias.")
    print("  B4 vs B5 asks whether the RIGHT person's history matters.")
    print("  If B4 does not beat both B2 and B5, do not oversell latent g.")
    print("\nWrote:", RESULT_ROOT.resolve())


if __name__ == "__main__":
    main()
