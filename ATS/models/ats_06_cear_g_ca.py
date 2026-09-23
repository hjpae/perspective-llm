"""
ATS 06 — CEAR-style recurrent perspective latent g on Community Alignment
==========================================================================

Purpose
-------
First serious technology-transfer test of the CEAR perspective architecture in a
language-based human-modeling domain.

Core architectural invariant
----------------------------

    g_(t-1)  ->  organization of current semantic evidence z_t
       ^                                                   |
       |                                                   v
       +--------- adaptive update from observed human evidence

This is NOT a static user embedding and NOT a DeepSets/history-average model.
The current implementation preserves the CEAR-style bidirectional loop:

    frozen text representation z_raw,t
        -> g-conditioned perspective coupler
        -> organized representation z_org,t
        -> predict human preference
        -> observe human choice / prediction error
        -> adaptive candidate update h_t and plasticity alpha_t
        -> g_t

Default perspective coupler
---------------------------
A state-conditioned LOW-RANK POSITIVE-DEFINITE METRIC over the *full* frozen
semantic embedding space. No dimensionality reduction is applied before the
metric:

    M_g = D_g + U_g U_g^T

where D_g is positive diagonal and U_g has small rank r << d.  We never build
M_g explicitly; M_g z is computed in O(d r).

At g=0 the metric is exactly identity, because the metric hypernetwork has no
bias. Thus resetting g cleanly removes person-state modulation while keeping the
same base predictor.

Important dataset limitation
----------------------------
Community Alignment supplies repeated-person preference evidence but NO trusted
chronology across conversations. Therefore:

* the recurrent architecture is retained;
* support conversations are RANDOMLY PERMUTED during training;
* validation/test predictions are averaged across multiple support permutations;
* no claim about real temporal value change is made from this experiment.

PersonaMem (or a future real longitudinal corpus) is the later temporal-revision
assay for the same updater.

Main test
---------
Does the RIGHT person's accumulated g improve held-out preference prediction?

    g_learned  vs  g_reset/query-only
    g_learned  vs  same-language wrong-person g

Primary metrics: NLL, Brier; secondary: accuracy.

Expected repo location
----------------------
perspective-llm/ATS/models/ats_06_cear_g_ca.py

Dependencies
------------
pip install datasets huggingface_hub sentence-transformers torch pandas numpy
"""

# %% Imports
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from huggingface_hub import get_token
from sentence_transformers import SentenceTransformer


# %% Paths / configuration
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]
CACHE_ROOT = REPO_ROOT / "data" / "ATS" / "cache" / "ca_cear_g"
RESULT_ROOT = REPO_ROOT / "results" / "ATS" / "cear_g_ca"

HF_DATASET = "facebook/community-alignment-dataset"
HF_TOKEN = os.environ.get("HF_TOKEN") or get_token()

SEED = 20260923
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Frozen multilingual semantic frontend.
TEXT_ENCODER = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBED_BATCH_SIZE = 128
PAIR_TEXT_MAX_CHARS = 12000  # only protects against pathological rows

# Controlled CA subset.
USE_WAVE = "1"
REQUIRE_PREGENERATED_FIRST_PROMPT = True
MIN_INTERACTIONS_PER_USER = 9

# User-disjoint split.
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
# remainder = test

# Support / query construction.
MIN_SUPPORT = 6
MIN_QUERY = 3
FIXED_SUPPORT_FRAC = 0.67
TRAIN_SUPPORT_MIN = 6
TRAIN_SUPPORT_MAX = 14
TRAIN_QUERIES_PER_USER = 2

# CEAR state.
G_DIM = 32
ALPHA_MIN = 0.03
ALPHA_MAX = 0.30
ALPHA_INIT_LOGIT = -0.75  # with tanh mapping -> alpha ~= .079

# Perspective coupling.
COUPLER = "metric"  # "metric" or "film"
METRIC_RANK = 8
METRIC_MAX_LOG_DIAG = 0.35
METRIC_U_SCALE = 0.15

# Small trainable heads; frozen semantic dimension itself is preserved.
BASE_HIDDEN = 128
UPDATE_OBS_DIM = 128
ERROR_EMBED_DIM = 32
ALPHA_HIDDEN = 96
DROPOUT = 0.10

# Training.
EPOCHS = 30
USERS_PER_BATCH = 32
LR = 3e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
PATIENCE = 5
SUPPORT_LOSS_WEIGHT = 0.25
G_NORM_WEIGHT = 1e-4

# Because CA has no trusted across-conversation order.
VAL_PERMUTATIONS = 3
TEST_PERMUTATIONS = 12
N_SWAP_REPEATS = 25

# Bootstrap.
N_BOOTSTRAP = 3000
BOOTSTRAP_SEED = 20260924

PROGRESS_EVERY = 5000
N_CHOICES = 4
POSITIONS = ("a", "b", "c", "d")
EPS = 1e-8

# Debug switch. Keeps architecture identical but shortens a run.
DEBUG = False
DEBUG_MAX_USERS_PER_SPLIT = 120
DEBUG_EPOCHS = 3


# %% Reproducibility
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


seed_everything(SEED)


# %% Text utilities
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
    return re.sub(r"\s+", " ", s).strip()


def stable_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


def as_bool(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    return clean_text(x).lower() in {"1", "true", "t", "yes", "y"}


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


def pair_text(prompt: str, response: str) -> str:
    """Frozen encoder input for one prompt/response candidate pair."""
    s = f"Prompt:\n{prompt}\n\nCandidate response:\n{response}"
    return s[:PAIR_TEXT_MAX_CHARS]


# %% Canonical candidate ordering
def canonicalize_row(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Remove A/B/C/D as a target shortcut by ordering candidates by content hash."""
    uid = clean_text(row.get("annotator_id"))
    lang = clean_text(row.get("assigned_lang"))
    prompt = normalize_text(row.get("first_turn_prompt"))
    raw_choice = normalize_choice(row.get("first_turn_preferred_response"))

    if not uid or not lang or not prompt or not raw_choice:
        return None

    candidates_by_pos = {
        p: normalize_text(row.get(f"first_turn_response_{p}"))
        for p in POSITIONS
    }
    if not all(candidates_by_pos.values()):
        return None

    hash_by_pos = {p: stable_hash(candidates_by_pos[p]) for p in POSITIONS}
    if len(set(hash_by_pos.values())) != N_CHOICES:
        return {"duplicate_candidates": True}

    ordered = sorted(
        [(hash_by_pos[p], candidates_by_pos[p]) for p in POSITIONS],
        key=lambda x: x[0],
    )
    hashes = [x[0] for x in ordered]
    candidates = [x[1] for x in ordered]
    chosen_hash = hash_by_pos[raw_choice]
    canonical_choice = hashes.index(chosen_hash)

    item_id = stable_hash(stable_hash(prompt) + "|" + "|".join(hashes))

    return {
        "duplicate_candidates": False,
        "user_id": uid,
        "language": lang,
        "item_id": item_id,
        "choice": int(canonical_choice),
        "prompt": prompt,
        "candidates": candidates,
    }


# %% Stream + compact cache
def build_or_load_compact_data() -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    interaction_path = CACHE_ROOT / "interactions.csv"
    item_path = CACHE_ROOT / "items.jsonl"

    if interaction_path.exists() and item_path.exists():
        print("Loading cached compact CA table...")
        df = pd.read_csv(interaction_path)
        items = {}
        with open(item_path, "r", encoding="utf-8") as f:
            for line in f:
                x = json.loads(line)
                items[x["item_id"]] = x
        return df, items

    print("Streaming Community Alignment from Hugging Face...")
    print("Raw corpus is NOT saved locally.\n")

    ds = load_dataset(
        HF_DATASET,
        split="train",
        streaming=True,
        token=HF_TOKEN,
    )

    rows: List[Dict[str, Any]] = []
    items: Dict[str, Dict[str, Any]] = {}
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

        rec = canonicalize_row(row)
        if rec is None:
            skipped["missing_fields"] += 1
            continue
        if rec["duplicate_candidates"]:
            skipped["duplicate_candidates"] += 1
            continue

        rows.append({
            "user_id": rec["user_id"],
            "language": rec["language"],
            "item_id": rec["item_id"],
            "choice": rec["choice"],
        })
        items.setdefault(rec["item_id"], {
            "item_id": rec["item_id"],
            "prompt": rec["prompt"],
            "candidates": rec["candidates"],
        })

        if i % PROGRESS_EVERY == 0:
            print(
                f"  streamed={i:,} | interactions={len(rows):,} | "
                f"unique items={len(items):,}"
            )

    df = pd.DataFrame(rows)

    # Same exact comparison encountered repeatedly by one user:
    # collapse if same choice, drop if conflicting.
    clean_rows = []
    same_choice_collapsed = 0
    conflict_pairs = 0
    for _, g in df.groupby(["user_id", "item_id"], sort=False):
        choices = g["choice"].unique()
        if len(choices) > 1:
            conflict_pairs += 1
            continue
        same_choice_collapsed += max(0, len(g) - 1)
        clean_rows.append(g.iloc[0])
    df = pd.DataFrame(clean_rows).reset_index(drop=True)

    # Drop rare users whose language label changes.
    n_lang = df.groupby("user_id")["language"].nunique()
    bad_users = set(n_lang[n_lang > 1].index.astype(str))
    if bad_users:
        df = df[~df["user_id"].isin(bad_users)].copy()

    # Need enough observations for support and held-out query.
    user_n = df.groupby("user_id").size()
    keep_users = set(
        user_n[user_n >= MIN_INTERACTIONS_PER_USER].index.astype(str)
    )
    df = df[df["user_id"].isin(keep_users)].reset_index(drop=True)

    used_items = set(df["item_id"])
    items = {k: v for k, v in items.items() if k in used_items}

    df.to_csv(interaction_path, index=False)
    with open(item_path, "w", encoding="utf-8") as f:
        for x in items.values():
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

    print("\n[compact CA cache]")
    print("interactions:", f"{len(df):,}")
    print("users:", f"{df['user_id'].nunique():,}")
    print("items:", f"{len(items):,}")
    print("same-choice duplicate rows collapsed:", same_choice_collapsed)
    print("conflicting user-item pairs dropped:", conflict_pairs)
    print("multi-language users dropped:", len(bad_users))
    print("other skips:", dict(skipped))

    return df, items


# %% User-disjoint split
def stratified_user_split(df: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    user_lang = df.groupby("user_id")["language"].first().reset_index()

    rows = []
    for lang, g in user_lang.groupby("language"):
        users = g["user_id"].astype(str).to_numpy()
        rng.shuffle(users)

        n = len(users)
        n_train = int(round(TRAIN_FRAC * n))
        n_val = int(round(VAL_FRAC * n))
        n_train = min(n_train, n)
        n_val = min(n_val, n - n_train)

        rows += [(u, lang, "train") for u in users[:n_train]]
        rows += [(u, lang, "val") for u in users[n_train:n_train+n_val]]
        rows += [(u, lang, "test") for u in users[n_train+n_val:]]

    return pd.DataFrame(rows, columns=["user_id", "language", "split"])


# %% Frozen full-dimensional pair embedding cache
def build_or_load_embeddings(
    items: Dict[str, Dict[str, Any]],
) -> Tuple[np.ndarray, Dict[str, int], Dict[str, int]]:
    """Encode prompt+candidate pairs; no projection before perspective metric."""
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    emb_path = CACHE_ROOT / "pair_embeddings.npy"
    item_index_path = CACHE_ROOT / "item_index.json"
    meta_path = CACHE_ROOT / "embedding_meta.json"

    if emb_path.exists() and item_index_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("encoder") == TEXT_ENCODER:
            print("Loading cached frozen pair embeddings...")
            E = np.load(emb_path)
            item_to_idx = json.loads(item_index_path.read_text(encoding="utf-8"))
            return E, item_to_idx, meta

    print(f"\nLoading frozen encoder: {TEXT_ENCODER}")
    encoder = SentenceTransformer(TEXT_ENCODER, device=DEVICE)

    item_ids = sorted(items)
    texts = []
    for item_id in item_ids:
        x = items[item_id]
        for c in x["candidates"]:
            texts.append(pair_text(x["prompt"], c))

    print("pair texts to embed:", f"{len(texts):,}")
    E2 = encoder.encode(
        texts,
        batch_size=EMBED_BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float16)

    embed_dim = int(E2.shape[1])
    E = E2.reshape(len(item_ids), N_CHOICES, embed_dim)
    item_to_idx = {item_id: i for i, item_id in enumerate(item_ids)}

    np.save(emb_path, E)
    item_index_path.write_text(json.dumps(item_to_idx), encoding="utf-8")
    meta = {
        "encoder": TEXT_ENCODER,
        "n_items": len(item_ids),
        "embed_dim": embed_dim,
        "normalized": True,
        "dtype": "float16",
        "pre_metric_projection": False,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("embedding tensor shape:", E.shape)
    print("embedding cache MB:", round(E.nbytes / 1024**2, 2))
    return E, item_to_idx, meta


# %% User histories
@dataclass
class UserHistory:
    user_id: str
    language: str
    items: np.ndarray
    choices: np.ndarray


@dataclass
class FixedEpisode:
    user_id: str
    language: str
    support_items: np.ndarray
    support_choices: np.ndarray
    query_items: np.ndarray
    query_choices: np.ndarray


def make_histories(
    df: pd.DataFrame,
    item_to_idx: Dict[str, int],
    split_df: pd.DataFrame,
) -> Dict[str, List[UserHistory]]:
    split_map = dict(zip(split_df["user_id"], split_df["split"]))
    out = {"train": [], "val": [], "test": []}

    for uid, g in df.groupby("user_id", sort=False):
        split = split_map.get(str(uid))
        if split is None:
            continue
        out[split].append(UserHistory(
            user_id=str(uid),
            language=str(g["language"].iloc[0]),
            items=np.asarray([item_to_idx[x] for x in g["item_id"]], dtype=np.int64),
            choices=g["choice"].to_numpy(dtype=np.int64),
        ))

    if DEBUG:
        for k in out:
            out[k] = out[k][:DEBUG_MAX_USERS_PER_SPLIT]
    return out


def deterministic_user_seed(uid: str, salt: int = 0) -> int:
    h = hashlib.sha1(f"{SEED}|{salt}|{uid}".encode()).hexdigest()[:8]
    return int(h, 16)


def make_fixed_episodes(histories: List[UserHistory]) -> List[FixedEpisode]:
    out = []
    for h in histories:
        n = len(h.items)
        if n < MIN_SUPPORT + MIN_QUERY:
            continue

        rng = np.random.default_rng(deterministic_user_seed(h.user_id))
        idx = np.arange(n)
        rng.shuffle(idx)

        n_support = max(MIN_SUPPORT, int(round(FIXED_SUPPORT_FRAC * n)))
        n_support = min(n_support, n - MIN_QUERY)
        s = idx[:n_support]
        q = idx[n_support:]

        out.append(FixedEpisode(
            user_id=h.user_id,
            language=h.language,
            support_items=h.items[s],
            support_choices=h.choices[s],
            query_items=h.items[q],
            query_choices=h.choices[q],
        ))
    return out


# %% Embedding store
class EmbeddingStore:
    def __init__(self, E: np.ndarray, device: str):
        # ~tens of MB for this subset; float32 simplifies training numerics.
        self.E = torch.tensor(E.astype(np.float32), dtype=torch.float32, device=device)
        self.embed_dim = int(E.shape[-1])

    def get(self, item_idx: Any) -> torch.Tensor:
        idx = torch.as_tensor(item_idx, dtype=torch.long, device=self.E.device)
        return self.E[idx]  # [..., 4, d]


# %% Perspective couplers
class LowRankMetricCoupler(nn.Module):
    """g-conditioned SPD metric over the full semantic embedding space.

    M_g = D_g + U_g U_g^T

    No bias in g->metric heads means g=0 -> D=I and U=0 exactly.
    """
    def __init__(self, embed_dim: int, g_dim: int, rank: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.g_dim = g_dim
        self.rank = rank

        self.diag_head = nn.Linear(g_dim, embed_dim, bias=False)
        self.u_head = nn.Linear(g_dim, embed_dim * rank, bias=False)

        # Start exactly at identity geometry.
        nn.init.zeros_(self.diag_head.weight)
        nn.init.zeros_(self.u_head.weight)

    def forward(self, z: torch.Tensor, g: torch.Tensor) -> Dict[str, torch.Tensor]:
        # z: [B, K, D], g: [B, G]
        B, K, D = z.shape

        log_diag = METRIC_MAX_LOG_DIAG * torch.tanh(self.diag_head(g))
        diag = torch.exp(log_diag)  # strictly positive, identity at g=0

        U = self.u_head(g).view(B, D, self.rank)
        U = (METRIC_U_SCALE / math.sqrt(self.rank)) * torch.tanh(U)

        # z @ U -> [B,K,R], then (... ) @ U^T -> [B,K,D]
        low_coords = torch.einsum("bkd,bdr->bkr", z, U)
        low_term = torch.einsum("bkr,bdr->bkd", low_coords, U)
        mz = diag[:, None, :] * z + low_term

        delta = mz - z
        quad_delta = z * delta
        energy = (z * mz).sum(dim=-1) / float(D)

        return {
            "z_org": mz,
            "delta": delta,
            "quad_delta": quad_delta,
            "energy": energy,
            "diag": diag,
            "u_norm": U.square().sum(dim=(1, 2)).sqrt(),
        }


class FiLMCoupler(nn.Module):
    """Historical CEAR coupling ablation; identity at g=0."""
    def __init__(self, embed_dim: int, g_dim: int):
        super().__init__()
        self.gamma = nn.Linear(g_dim, embed_dim, bias=False)
        self.beta = nn.Linear(g_dim, embed_dim, bias=False)
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.beta.weight)

    def forward(self, z: torch.Tensor, g: torch.Tensor) -> Dict[str, torch.Tensor]:
        gamma = 0.35 * torch.tanh(self.gamma(g))
        beta = 0.15 * torch.tanh(self.beta(g))
        z_org = (1.0 + gamma[:, None, :]) * z + beta[:, None, :]
        delta = z_org - z
        quad_delta = z * delta
        energy = (z * z_org).sum(dim=-1) / float(z.shape[-1])
        return {
            "z_org": z_org,
            "delta": delta,
            "quad_delta": quad_delta,
            "energy": energy,
            "diag": 1.0 + gamma,
            "u_norm": torch.zeros(z.shape[0], device=z.device),
        }


def make_coupler(embed_dim: int) -> nn.Module:
    if COUPLER == "metric":
        return LowRankMetricCoupler(embed_dim, G_DIM, METRIC_RANK)
    if COUPLER == "film":
        return FiLMCoupler(embed_dim, G_DIM)
    raise ValueError(f"Unknown COUPLER={COUPLER}")


# %% CEAR-style model
class CEARHumanStateModel(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.coupler = make_coupler(embed_dim)

        # Query-only semantic population scorer.
        self.base_scorer = nn.Sequential(
            nn.Linear(embed_dim, BASE_HIDDEN),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(BASE_HIDDEN, 1),
        )

        # g-dependent correction. Inputs are exactly zero at reset g=0.
        self.person_scorer = nn.Linear(embed_dim * 2, 1, bias=False)
        self.person_scale_log = nn.Parameter(torch.tensor(math.log(0.10)))

        # Current organized evidence -> recurrent candidate update.
        self.obs_projector = nn.Sequential(
            nn.Linear(embed_dim * 3, UPDATE_OBS_DIM),
            nn.LayerNorm(UPDATE_OBS_DIM),
            nn.GELU(),
        )
        self.error_projector = nn.Sequential(
            nn.Linear(6, ERROR_EMBED_DIM),
            nn.GELU(),
        )

        update_input_dim = UPDATE_OBS_DIM + ERROR_EMBED_DIM
        self.gru = nn.GRUCell(update_input_dim, G_DIM)
        self.candidate_ln = nn.LayerNorm(G_DIM)

        self.alpha_net = nn.Sequential(
            nn.Linear(update_input_dim + G_DIM, ALPHA_HIDDEN),
            nn.Tanh(),
            nn.Linear(ALPHA_HIDDEN, G_DIM),
        )
        # Start with same alpha for all dimensions, ~0.08, then learn state dependence.
        nn.init.zeros_(self.alpha_net[-1].weight)
        nn.init.constant_(self.alpha_net[-1].bias, ALPHA_INIT_LOGIT)

    def predict(self, z_raw: torch.Tensor, g_prev: torch.Tensor):
        """Predict one 4-way preference before observing the human choice."""
        geo = self.coupler(z_raw, g_prev)
        base = self.base_scorer(z_raw).squeeze(-1)

        # Full-dimensional geometric change, not a pre-metric bottleneck.
        person_feat = torch.cat([geo["delta"], geo["quad_delta"]], dim=-1)
        person = self.person_scorer(person_feat).squeeze(-1)
        scale = self.person_scale_log.exp().clamp(max=10.0)
        logits = base + scale * person
        return logits, geo

    def update_g(
        self,
        g_prev: torch.Tensor,
        z_raw: torch.Tensor,
        geo: Dict[str, torch.Tensor],
        choice: torch.Tensor,
        error_features: torch.Tensor,
    ):
        """Observed choice updates g after prediction (teacher-forced human evidence)."""
        B, K, D = z_raw.shape
        idx = choice[:, None, None].expand(B, 1, D)

        chosen_raw = z_raw.gather(1, idx).squeeze(1)
        chosen_org = geo["z_org"].gather(1, idx).squeeze(1)
        chosen_quad_delta = geo["quad_delta"].gather(1, idx).squeeze(1)

        obs = self.obs_projector(
            torch.cat([chosen_raw, chosen_org, chosen_quad_delta], dim=-1)
        )
        err = self.error_projector(error_features)
        update_input = torch.cat([obs, err], dim=-1)

        candidate = self.candidate_ln(self.gru(update_input, g_prev))
        alpha_logits = self.alpha_net(torch.cat([update_input, g_prev], dim=-1))

        alpha_bar = 0.5 * (ALPHA_MIN + ALPHA_MAX)
        alpha_half_range = 0.5 * (ALPHA_MAX - ALPHA_MIN)
        alpha = alpha_bar + alpha_half_range * torch.tanh(alpha_logits)

        g_new = (1.0 - alpha) * g_prev + alpha * candidate
        return g_new, alpha


# %% Error-state helper
@dataclass
class ErrorState:
    short: torch.Tensor
    long: torch.Tensor


def initial_error_state(batch_size: int, device: str) -> ErrorState:
    chance_nll = math.log(N_CHOICES)
    return ErrorState(
        short=torch.full((batch_size,), chance_nll, device=device),
        long=torch.full((batch_size,), chance_nll, device=device),
    )


def build_error_features(
    logits: torch.Tensor,
    choice: torch.Tensor,
    state: ErrorState,
) -> Tuple[torch.Tensor, ErrorState]:
    """Prediction-error features; detached before entering the state updater."""
    with torch.no_grad():
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        nll = -log_probs.gather(1, choice[:, None]).squeeze(1)
        p_true = probs.gather(1, choice[:, None]).squeeze(1)
        entropy = -(probs * torch.log(probs.clamp_min(EPS))).sum(dim=-1)
        top2 = torch.topk(probs, k=2, dim=-1).values
        margin = top2[:, 0] - top2[:, 1]

        feat = torch.stack([
            nll,
            state.short,
            state.long,
            nll - state.long,
            entropy,
            p_true - margin,
        ], dim=-1)

        new_state = ErrorState(
            short=0.50 * state.short + 0.50 * nll,
            long=0.90 * state.long + 0.10 * nll,
        )
    return feat, new_state


# %% Train episode sampling / batch assembly
def sample_train_episode(h: UserHistory, rng: np.random.Generator):
    n = len(h.items)
    max_support = min(TRAIN_SUPPORT_MAX, n - TRAIN_QUERIES_PER_USER)
    if max_support < TRAIN_SUPPORT_MIN:
        return None

    n_support = int(rng.integers(TRAIN_SUPPORT_MIN, max_support + 1))
    idx = np.arange(n)
    rng.shuffle(idx)

    s = idx[:n_support]  # random order = CA order augmentation
    q = idx[n_support:n_support + TRAIN_QUERIES_PER_USER]

    return h.items[s], h.choices[s], h.items[q], h.choices[q]


def make_train_batch(
    users: List[UserHistory],
    store: EmbeddingStore,
    rng: np.random.Generator,
):
    episodes = [sample_train_episode(h, rng) for h in users]
    episodes = [x for x in episodes if x is not None]
    if not episodes:
        return None

    B = len(episodes)
    S = max(len(x[0]) for x in episodes)
    Q = TRAIN_QUERIES_PER_USER

    s_items = np.zeros((B, S), dtype=np.int64)
    s_choices = np.zeros((B, S), dtype=np.int64)
    s_mask = np.zeros((B, S), dtype=bool)
    q_items = np.zeros((B, Q), dtype=np.int64)
    q_choices = np.zeros((B, Q), dtype=np.int64)

    for b, (si, sy, qi, qy) in enumerate(episodes):
        s_items[b, :len(si)] = si
        s_choices[b, :len(sy)] = sy
        s_mask[b, :len(si)] = True
        q_items[b, :len(qi)] = qi
        q_choices[b, :len(qy)] = qy

    return {
        "support_z": store.get(s_items),                 # B,S,4,D
        "support_y": torch.tensor(s_choices, dtype=torch.long, device=DEVICE),
        "support_mask": torch.tensor(s_mask, dtype=torch.bool, device=DEVICE),
        "query_z": store.get(q_items),                   # B,Q,4,D
        "query_y": torch.tensor(q_choices, dtype=torch.long, device=DEVICE),
    }


# %% Recurrent support pass
def run_support_batch(
    model: CEARHumanStateModel,
    support_z: torch.Tensor,
    support_y: torch.Tensor,
    support_mask: torch.Tensor,
    collect_stats: bool = False,
):
    B, S, K, D = support_z.shape
    g = torch.zeros((B, G_DIM), device=support_z.device)
    err_state = initial_error_state(B, support_z.device)

    support_ce_sum = torch.tensor(0.0, device=support_z.device)
    support_n = 0
    alpha_records = []

    for t in range(S):
        active = support_mask[:, t]
        if not active.any():
            continue

        z_t = support_z[:, t]
        y_t = support_y[:, t]

        logits, geo = model.predict(z_t, g)

        ce_each = F.cross_entropy(logits, y_t, reduction="none")
        support_ce_sum = support_ce_sum + ce_each[active].sum()
        support_n += int(active.sum().item())

        err_feat, new_err_state = build_error_features(logits, y_t, err_state)
        g_candidate, alpha = model.update_g(g, z_t, geo, y_t, err_feat)

        # Padded users retain old state.
        g = torch.where(active[:, None], g_candidate, g)
        err_state = ErrorState(
            short=torch.where(active, new_err_state.short, err_state.short),
            long=torch.where(active, new_err_state.long, err_state.long),
        )

        if collect_stats:
            alpha_records.append(alpha.detach())

    support_ce = support_ce_sum / max(1, support_n)

    if collect_stats and alpha_records:
        alpha_tensor = torch.stack(alpha_records, dim=1)  # B,S,G (contains padded-step values)
    else:
        alpha_tensor = None

    return g, support_ce, alpha_tensor


def query_loss(
    model: CEARHumanStateModel,
    query_z: torch.Tensor,
    query_y: torch.Tensor,
    g: torch.Tensor,
):
    B, Q, K, D = query_z.shape
    z = query_z.reshape(B * Q, K, D)
    y = query_y.reshape(B * Q)
    gq = g[:, None, :].expand(B, Q, G_DIM).reshape(B * Q, G_DIM)
    logits, _ = model.predict(z, gq)
    return F.cross_entropy(logits, y), logits


# %% Training
def evaluate_validation_nll(
    model: CEARHumanStateModel,
    episodes: List[FixedEpisode],
    store: EmbeddingStore,
) -> float:
    df, _, _ = evaluate_episodes(
        model,
        episodes,
        store,
        n_permutations=VAL_PERMUTATIONS,
        mode="learned",
        seed_base=SEED + 3000,
    )
    return float(df["nll"].mean())


def train_model(
    model: CEARHumanStateModel,
    train_histories: List[UserHistory],
    val_episodes: List[FixedEpisode],
    store: EmbeddingStore,
):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )

    rng = np.random.default_rng(SEED + 100)
    n_epochs = DEBUG_EPOCHS if DEBUG else EPOCHS

    best_val = float("inf")
    best_state = None
    bad_epochs = 0
    hist = []

    for epoch in range(1, n_epochs + 1):
        model.train()
        order = np.arange(len(train_histories))
        rng.shuffle(order)

        losses, q_losses, s_losses, g_norms = [], [], [], []

        for start in range(0, len(order), USERS_PER_BATCH):
            idx = order[start:start + USERS_PER_BATCH]
            users = [train_histories[i] for i in idx]
            batch = make_train_batch(users, store, rng)
            if batch is None:
                continue

            g, support_ce, _ = run_support_batch(
                model,
                batch["support_z"],
                batch["support_y"],
                batch["support_mask"],
                collect_stats=False,
            )
            q_ce, _ = query_loss(
                model, batch["query_z"], batch["query_y"], g
            )

            g_norm = g.square().mean()
            loss = q_ce + SUPPORT_LOSS_WEIGHT * support_ce + G_NORM_WEIGHT * g_norm

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            losses.append(float(loss.item()))
            q_losses.append(float(q_ce.item()))
            s_losses.append(float(support_ce.item()))
            g_norms.append(float(g_norm.item()))

        model.eval()
        val_nll = evaluate_validation_nll(model, val_episodes, store)

        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "train_query_ce": float(np.mean(q_losses)),
            "train_support_ce": float(np.mean(s_losses)),
            "train_g_norm": float(np.mean(g_norms)),
            "val_nll": val_nll,
            "person_scale": float(model.person_scale_log.exp().item()),
        }
        hist.append(row)

        print(
            f"epoch {epoch:02d} | trainQ={row['train_query_ce']:.4f} | "
            f"trainS={row['train_support_ce']:.4f} | val NLL={val_nll:.4f} | "
            f"person_scale={row['person_scale']:.3f}"
        )

        if val_nll < best_val - 1e-4:
            best_val = val_nll
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= PATIENCE:
            print("early stopping")
            break

    if best_state is None:
        raise RuntimeError("No valid checkpoint was produced.")
    model.load_state_dict(best_state)
    return pd.DataFrame(hist), best_val


# %% Fixed episode helper
def make_single_support_tensor(
    ep: FixedEpisode,
    store: EmbeddingStore,
    permutation: np.ndarray,
):
    items = ep.support_items[permutation]
    choices = ep.support_choices[permutation]
    z = store.get(items)[None, ...]  # 1,S,4,D
    y = torch.tensor(choices[None, :], dtype=torch.long, device=DEVICE)
    mask = torch.ones((1, len(items)), dtype=torch.bool, device=DEVICE)
    return z, y, mask


@torch.no_grad()
def infer_g_for_episode(
    model: CEARHumanStateModel,
    ep: FixedEpisode,
    store: EmbeddingStore,
    permutation: np.ndarray,
):
    sz, sy, sm = make_single_support_tensor(ep, store, permutation)
    g, _, alphas = run_support_batch(
        model, sz, sy, sm, collect_stats=True
    )
    return g.squeeze(0), alphas.squeeze(0) if alphas is not None else None


@torch.no_grad()
def predict_episode_queries(
    model: CEARHumanStateModel,
    ep: FixedEpisode,
    store: EmbeddingStore,
    g: torch.Tensor,
):
    z = store.get(ep.query_items)  # Q,4,D
    gq = g[None, :].expand(len(ep.query_items), -1)
    logits, _ = model.predict(z, gq)
    return F.softmax(logits, dim=-1)


# %% Evaluation with permutation averaging
@torch.no_grad()
def evaluate_episodes(
    model: CEARHumanStateModel,
    episodes: List[FixedEpisode],
    store: EmbeddingStore,
    n_permutations: int,
    mode: str,
    seed_base: int,
    donor_map_by_repeat: Optional[List[Dict[str, str]]] = None,
    cached_g_by_repeat: Optional[List[Dict[str, torch.Tensor]]] = None,
):
    """Average query probability over random support permutations.

    mode:
      learned : own g from own support
      reset   : zero g (same model; metric identity)
      swap    : donor user's g, same language, per repeat
    """
    model.eval()

    prob_accum: Dict[str, np.ndarray] = {}
    true_map: Dict[str, np.ndarray] = {}
    g_records: Dict[str, List[np.ndarray]] = defaultdict(list)
    alpha_rows = []

    # reset has no reason to repeat support passes
    repeats = 1 if mode == "reset" else n_permutations

    own_g_cache: List[Dict[str, torch.Tensor]] = []

    for rep in range(repeats):
        rep_g: Dict[str, torch.Tensor] = {}

        if mode in {"learned", "swap"} and cached_g_by_repeat is None:
            for ep in episodes:
                rng = np.random.default_rng(
                    deterministic_user_seed(ep.user_id, salt=seed_base + rep)
                )
                perm = rng.permutation(len(ep.support_items))
                g, alphas = infer_g_for_episode(model, ep, store, perm)
                rep_g[ep.user_id] = g
                g_records[ep.user_id].append(g.detach().cpu().numpy())

                if alphas is not None:
                    alpha_rows.append({
                        "repeat": rep,
                        "user_id": ep.user_id,
                        "language": ep.language,
                        "alpha_mean": float(alphas.mean().item()),
                        "alpha_sd": float(alphas.std().item()),
                        "alpha_min": float(alphas.min().item()),
                        "alpha_max": float(alphas.max().item()),
                    })
            own_g_cache.append(rep_g)
        elif cached_g_by_repeat is not None:
            rep_g = cached_g_by_repeat[rep]
            own_g_cache.append(rep_g)

        for ep in episodes:
            if mode == "reset":
                g = torch.zeros(G_DIM, device=DEVICE)
            elif mode == "learned":
                g = rep_g[ep.user_id]
            elif mode == "swap":
                if donor_map_by_repeat is None:
                    raise ValueError("swap mode requires donor_map_by_repeat")
                donor = donor_map_by_repeat[rep][ep.user_id]
                g = rep_g[donor]
            else:
                raise ValueError(mode)

            probs = predict_episode_queries(model, ep, store, g).cpu().numpy()
            if ep.user_id not in prob_accum:
                prob_accum[ep.user_id] = np.zeros_like(probs, dtype=np.float64)
                true_map[ep.user_id] = ep.query_choices.copy()
            prob_accum[ep.user_id] += probs / repeats

    rows = []
    for ep in episodes:
        P = prob_accum[ep.user_id]
        Y = true_map[ep.user_id]
        pred = P.argmax(axis=1)
        for j, y in enumerate(Y):
            onehot = np.zeros(N_CHOICES, dtype=float)
            onehot[int(y)] = 1.0
            rows.append({
                "user_id": ep.user_id,
                "language": ep.language,
                "mode": mode,
                "query_index": j,
                "correct": int(pred[j] == int(y)),
                "nll": float(-math.log(max(EPS, P[j, int(y)]))),
                "brier": float(np.square(P[j] - onehot).sum()),
                "p_true": float(P[j, int(y)]),
            })

    # Order sensitivity: cosine of each g to that user's permutation-mean g.
    stability_rows = []
    if g_records:
        for uid, gs in g_records.items():
            A = np.asarray(gs, dtype=float)
            mean_g = A.mean(axis=0)
            mean_norm = np.linalg.norm(mean_g) + EPS
            cos = []
            for x in A:
                cos.append(float(np.dot(x, mean_g) / ((np.linalg.norm(x) + EPS) * mean_norm)))
            stability_rows.append({
                "user_id": uid,
                "n_permutations": len(gs),
                "mean_cosine_to_perm_mean": float(np.mean(cos)),
                "min_cosine_to_perm_mean": float(np.min(cos)),
                "g_norm_perm_mean": float(np.linalg.norm(mean_g)),
                "g_norm_across_perm_sd": float(np.std(np.linalg.norm(A, axis=1))),
            })

    return (
        pd.DataFrame(rows),
        own_g_cache,
        {
            "stability": pd.DataFrame(stability_rows),
            "alpha": pd.DataFrame(alpha_rows),
        },
    )


def make_same_language_donor_maps(
    episodes: List[FixedEpisode],
    n_repeats: int,
    seed_base: int,
) -> List[Dict[str, str]]:
    by_lang = defaultdict(list)
    for ep in episodes:
        by_lang[ep.language].append(ep.user_id)

    maps = []
    for rep in range(n_repeats):
        rng = np.random.default_rng(seed_base + rep)
        m = {}
        for lang, users0 in by_lang.items():
            users = list(users0)
            if len(users) < 2:
                for u in users:
                    m[u] = u
                continue

            donors = users.copy()
            # deterministic derangement by random cyclic shift
            rng.shuffle(donors)
            shift = int(rng.integers(1, len(donors)))
            donors = donors[shift:] + donors[:shift]
            # repair accidental fixed points by simple swaps
            for i in range(len(users)):
                if donors[i] == users[i]:
                    j = (i + 1) % len(users)
                    donors[i], donors[j] = donors[j], donors[i]
            for u, d in zip(users, donors):
                if d == u:
                    # extremely rare pathological repair; choose next user.
                    d = users[(users.index(u) + 1) % len(users)]
                m[u] = d
        maps.append(m)
    return maps


# %% Metrics / bootstrap
def summarize_predictions(df: pd.DataFrame) -> Dict[str, Any]:
    return {
        "n_query": int(len(df)),
        "n_users": int(df["user_id"].nunique()),
        "accuracy": float(df["correct"].mean()),
        "nll": float(df["nll"].mean()),
        "brier": float(df["brier"].mean()),
        "mean_p_true": float(df["p_true"].mean()),
    }


def per_user_metric(df: pd.DataFrame, metric: str) -> pd.Series:
    col = "correct" if metric == "accuracy" else metric
    return df.groupby("user_id")[col].mean()


def bootstrap_delta(
    a: pd.DataFrame,
    b: pd.DataFrame,
    metric: str,
    name_a: str,
    name_b: str,
    seed_offset: int = 0,
) -> Dict[str, Any]:
    xa = per_user_metric(a, metric)
    xb = per_user_metric(b, metric)
    common = xa.index.intersection(xb.index)
    delta = (xa.loc[common] - xb.loc[common]).to_numpy()

    rng = np.random.default_rng(BOOTSTRAP_SEED + seed_offset)
    boots = np.empty(N_BOOTSTRAP, dtype=float)
    n = len(delta)
    for i in range(N_BOOTSTRAP):
        ix = rng.integers(0, n, size=n)
        boots[i] = delta[ix].mean()

    return {
        "model_a": name_a,
        "model_b": name_b,
        "metric": metric,
        "n_users": int(n),
        "delta": float(delta.mean()),
        "ci_low": float(np.quantile(boots, 0.025)),
        "ci_high": float(np.quantile(boots, 0.975)),
    }


# %% Save mean g across support permutations
@torch.no_grad()
def save_test_g_states(
    test_episodes: List[FixedEpisode],
    g_cache: List[Dict[str, torch.Tensor]],
) -> pd.DataFrame:
    ep_map = {ep.user_id: ep for ep in test_episodes}
    rows = []
    for uid in ep_map:
        A = np.stack([
            rep[uid].detach().cpu().numpy() for rep in g_cache
        ], axis=0)
        mean_g = A.mean(axis=0)
        row = {
            "user_id": uid,
            "language": ep_map[uid].language,
            "support_n": len(ep_map[uid].support_items),
            "query_n": len(ep_map[uid].query_items),
        }
        row.update({f"g_{j:02d}": float(v) for j, v in enumerate(mean_g)})
        rows.append(row)
    return pd.DataFrame(rows)


# %% Main
def main():
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)

    print("REPO_ROOT:  ", REPO_ROOT)
    print("CACHE_ROOT: ", CACHE_ROOT)
    print("RESULT_ROOT:", RESULT_ROOT)
    print("DEVICE:     ", DEVICE)
    print("COUPLER:    ", COUPLER)
    print("HF auth:    ", "yes" if HF_TOKEN else "no (public streaming)")
    print()

    # Data / frozen semantics.
    df, items = build_or_load_compact_data()
    split_df = stratified_user_split(df)
    split_df.to_csv(RESULT_ROOT / "user_splits.csv", index=False)

    E, item_to_idx, emb_meta = build_or_load_embeddings(items)
    store = EmbeddingStore(E, DEVICE)
    histories = make_histories(df, item_to_idx, split_df)

    val_eps = make_fixed_episodes(histories["val"])
    test_eps = make_fixed_episodes(histories["test"])

    print("\n[user coverage]")
    for split in ("train", "val", "test"):
        lens = [len(h.items) for h in histories[split]]
        print(
            f"{split:5s} users={len(lens):4d} | "
            f"median interactions={np.median(lens) if lens else np.nan:.1f}"
        )
    print("fixed val users:", len(val_eps))
    print("fixed test users:", len(test_eps))

    # Model.
    model = CEARHumanStateModel(store.embed_dim).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("\n[model]")
    print("frozen semantic dimension:", store.embed_dim)
    print("pre-metric dimensionality reduction: NO")
    print("g dimension:", G_DIM)
    print("g storage/user float32:", G_DIM * 4, "bytes")
    if COUPLER == "metric":
        print("metric rank:", METRIC_RANK)
    print("trainable parameters:", f"{n_params:,}")

    # Train.
    history_df, best_val = train_model(
        model, histories["train"], val_eps, store
    )
    history_df.to_csv(RESULT_ROOT / "training_history.csv", index=False)

    torch.save({
        "state_dict": model.state_dict(),
        "config": {
            "coupler": COUPLER,
            "text_encoder": TEXT_ENCODER,
            "embed_dim": store.embed_dim,
            "g_dim": G_DIM,
            "metric_rank": METRIC_RANK if COUPLER == "metric" else None,
            "alpha_min": ALPHA_MIN,
            "alpha_max": ALPHA_MAX,
            "seed": SEED,
        },
    }, RESULT_ROOT / "cear_g_model.pt")

    # TEST: own g, reset g.
    learned_df, learned_g_cache, learned_aux = evaluate_episodes(
        model,
        test_eps,
        store,
        n_permutations=TEST_PERMUTATIONS,
        mode="learned",
        seed_base=SEED + 10000,
    )
    reset_df, _, _ = evaluate_episodes(
        model,
        test_eps,
        store,
        n_permutations=1,
        mode="reset",
        seed_base=SEED + 20000,
    )

    # Same-language wrong-person state, using each repeat's actual inferred g.
    donor_maps = make_same_language_donor_maps(
        test_eps, TEST_PERMUTATIONS, SEED + 30000
    )
    swapped_df, _, _ = evaluate_episodes(
        model,
        test_eps,
        store,
        n_permutations=TEST_PERMUTATIONS,
        mode="swap",
        seed_base=SEED + 10000,
        donor_map_by_repeat=donor_maps,
        cached_g_by_repeat=learned_g_cache,
    )

    # Summaries.
    rows = []
    for name, d in [
        ("g_learned", learned_df),
        ("g_reset_query_only", reset_df),
        ("g_swapped_same_language", swapped_df),
    ]:
        r = summarize_predictions(d)
        r["model"] = name
        rows.append(r)
    summary_df = pd.DataFrame(rows).sort_values("nll")
    summary_df.to_csv(RESULT_ROOT / "test_summary.csv", index=False)

    deltas = []
    seed_off = 0
    for metric in ("accuracy", "nll", "brier"):
        deltas.append(bootstrap_delta(
            learned_df, reset_df, metric,
            "g_learned", "g_reset_query_only", seed_off
        ))
        seed_off += 1
        deltas.append(bootstrap_delta(
            learned_df, swapped_df, metric,
            "g_learned", "g_swapped_same_language", seed_off
        ))
        seed_off += 1
    delta_df = pd.DataFrame(deltas)
    delta_df.to_csv(RESULT_ROOT / "bootstrap_deltas.csv", index=False)

    # Diagnostics specific to CEAR transfer.
    learned_aux["stability"].to_csv(
        RESULT_ROOT / "support_order_stability.csv", index=False
    )
    learned_aux["alpha"].to_csv(
        RESULT_ROOT / "alpha_diagnostics.csv", index=False
    )
    save_test_g_states(test_eps, learned_g_cache).to_csv(
        RESULT_ROOT / "test_g_states.csv", index=False
    )

    learned_df.to_csv(
        RESULT_ROOT / "test_predictions_g_learned.csv", index=False
    )
    reset_df.to_csv(
        RESULT_ROOT / "test_predictions_g_reset.csv", index=False
    )
    swapped_df.to_csv(
        RESULT_ROOT / "test_predictions_g_swapped.csv", index=False
    )

    metadata = {
        "dataset": HF_DATASET,
        "wave": USE_WAVE,
        "pregenerated_first_prompt_only": REQUIRE_PREGENERATED_FIRST_PROMPT,
        "dataset_is_longitudinal": False,
        "support_order_semantics": "unknown; randomized in training and averaged at evaluation",
        "text_encoder": TEXT_ENCODER,
        "text_encoder_frozen": True,
        "semantic_embed_dim": store.embed_dim,
        "pre_metric_projection": False,
        "coupler": COUPLER,
        "metric_rank": METRIC_RANK if COUPLER == "metric" else None,
        "g_dim": G_DIM,
        "g_state_bytes_float32": G_DIM * 4,
        "alpha_min": ALPHA_MIN,
        "alpha_max": ALPHA_MAX,
        "trainable_params": n_params,
        "best_val_nll": best_val,
        "test_permutations": TEST_PERMUTATIONS,
        "device": DEVICE,
        "seed": SEED,
        "embedding_meta": emb_meta,
    }
    (RESULT_ROOT / "experiment_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    # Console.
    print("\n" + "=" * 84)
    print("ATS 06 — CEAR-STYLE RECURRENT g — TEST")
    print("=" * 84)
    print(summary_df.to_string(index=False))

    print("\n[user-cluster bootstrap deltas: model_a - model_b]")
    print(delta_df.to_string(index=False))

    if not learned_aux["stability"].empty:
        st = learned_aux["stability"]
        print("\n[support-order robustness]")
        print("median cosine(g_perm, mean_g):", round(float(st["mean_cosine_to_perm_mean"].median()), 4))
        print("median minimum cosine:        ", round(float(st["min_cosine_to_perm_mean"].median()), 4))

    if not learned_aux["alpha"].empty:
        aa = learned_aux["alpha"]
        print("\n[adaptive plasticity]")
        print("mean alpha:", round(float(aa["alpha_mean"].mean()), 4))
        print("mean within-user alpha SD:", round(float(aa["alpha_sd"].mean()), 4))

    print("\nInterpretation:")
    print("  Accuracy: positive g_learned - baseline is good.")
    print("  NLL/Brier: negative g_learned - baseline is good.")
    print("  learned < reset NLL => accumulated state adds predictive value.")
    print("  learned < swapped NLL => RIGHT-person state matters.")
    print("  Order-stability is a CA-specific diagnostic, not a temporal claim.")
    print("\nWrote:", RESULT_ROOT.resolve())


if __name__ == "__main__":
    main()
