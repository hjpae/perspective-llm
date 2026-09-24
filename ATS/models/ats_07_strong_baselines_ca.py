"""
ATS 07 — Strong baseline challenge on Community Alignment
==========================================================

Goal
----
Keep the validated ATS06 CEAR-style recurrent g model FIXED and ask a narrower
question:

    Does recurrent person-state g add value beyond simpler history summaries?

This script compares, on the exact ATS06 seed-0 user split and fixed val/test
support-query episodes:

    1) ATS reset / query-only         : g = 0 in the validated ATS06 model
    2) Static profile                 : order-invariant mean preference evidence
    3) Semantic retrieval            : query-conditioned top-k support evidence
    4) ATS g                          : validated recurrent right-person state
    5) ATS + retrieval               : validation-tuned log-probability fusion

Important scope note
--------------------
"Semantic retrieval" here is a compact RAG-style baseline over the same frozen
semantic embeddings. It is NOT yet a full production LLM+RAG system. Its role is
to test whether query-local access to support history explains the ATS gain.

The ATS06 architecture/checkpoint is NOT retrained here.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Defaults
# -----------------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]

SEED = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EPOCHS = 120
USERS_PER_BATCH = 32
LR = 3e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
PATIENCE = 20
MIN_DELTA = 1e-4
DROPOUT = 0.10
HIDDEN = 128

RETRIEVAL_K_CHOICES = (3, 5, 10)
RETRIEVAL_TEMPERATURE = 0.20

VAL_ATS_PERMUTATIONS = 3
TEST_ATS_PERMUTATIONS = 12
N_BOOTSTRAP = 3000
BOOTSTRAP_SEED = 17
EPS = 1e-8


# -----------------------------------------------------------------------------
# CLI / paths
# -----------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="ATS07: static-profile and semantic-retrieval challenge for ATS06.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ats06-dir", required=True,
                   help="ATS06 seed-0 result directory containing best.pt, metadata, user_splits.csv.")
    p.add_argument("--ats06-source", default=None,
                   help="Path to ATS06 Python source. If omitted, common sibling names are tried.")
    p.add_argument("--result-dir", required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=SEED,
                   help="Training seed for ATS07 baselines. Split/episode seed remains fixed from ATS06.")
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--users-per-batch", type=int, default=USERS_PER_BATCH)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    p.add_argument("--patience", type=int, default=PATIENCE)
    p.add_argument("--min-delta", type=float, default=MIN_DELTA)
    p.add_argument("--retrieval-k", default="3,5,10",
                   help="Comma-separated k values used as train augmentation and validation sweep.")
    p.add_argument("--retrieval-temperature", type=float, default=RETRIEVAL_TEMPERATURE)
    p.add_argument("--val-ats-permutations", type=int, default=VAL_ATS_PERMUTATIONS)
    p.add_argument("--test-ats-permutations", type=int, default=TEST_ATS_PERMUTATIONS)
    p.add_argument("--bootstrap", type=int, default=N_BOOTSTRAP)
    p.add_argument("--debug", action="store_true")
    return p


def resolve_repo_path(value: str) -> Path:
    p = Path(value).expanduser()
    return p.resolve() if p.is_absolute() else (REPO_ROOT / p).resolve()


def resolve_device(value: str) -> str:
    v = value.lower().strip()
    if v == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if v.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {value!r}, but CUDA is unavailable.")
    return value


def parse_k_choices(text: str) -> Tuple[int, ...]:
    vals = tuple(sorted({int(x.strip()) for x in text.split(",") if x.strip()}))
    if not vals or min(vals) < 1:
        raise ValueError("--retrieval-k must contain positive integers")
    return vals


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----------------------------------------------------------------------------
# Load ATS06 as a module so the exact dataset/cache/CEAR implementation is reused
# -----------------------------------------------------------------------------
def discover_ats06_source(user_value: Optional[str]) -> Path:
    if user_value:
        p = resolve_repo_path(user_value)
        if not p.exists():
            raise FileNotFoundError(p)
        return p

    candidates = [
        REPO_ROOT / "ATS" / "models" / "ats_06_cear_g_ca.py",
        REPO_ROOT / "ATS" / "models" / "ats_06_cear_g_ca_replication.py",
    ]
    for p in candidates:
        if p.exists():
            return p.resolve()
    raise FileNotFoundError(
        "Could not find ATS06 source. Pass --ats06-source explicitly. Tried:\n" +
        "\n".join(str(x) for x in candidates)
    )


def load_module_from_path(path: Path):
    spec = importlib.util.spec_from_file_location("ats06_runtime", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import ATS06 source: {path}")
    mod = importlib.util.module_from_spec(spec)
    # dataclasses (Python 3.11+) expects the module to already exist in sys.modules
    # while class decorators are executed during dynamic import.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def configure_ats06_module(ats06, metadata: Dict[str, Any], device: str, episode_seed: int) -> None:
    # Fixed split is loaded from CSV. The episode seed reconstructs the same fixed
    # support/query partition inside val/test users.
    ats06.SEED = int(episode_seed)
    ats06.DEVICE = device
    ats06.COUPLER = str(metadata["coupler"])
    ats06.G_DIM = int(metadata["g_dim"])
    if metadata.get("metric_rank") is not None:
        ats06.METRIC_RANK = int(metadata["metric_rank"])
    ats06.ALPHA_MIN = float(metadata["alpha_min"])
    ats06.ALPHA_MAX = float(metadata["alpha_max"])
    ats06.TEXT_ENCODER = str(metadata["text_encoder"])
    ats06.seed_everything(int(episode_seed))


# -----------------------------------------------------------------------------
# History evidence and baselines
# -----------------------------------------------------------------------------
def chosen_preference_evidence(
    support_z: torch.Tensor,
    support_y: torch.Tensor,
) -> torch.Tensor:
    """Chosen candidate minus mean of the three unchosen candidates.

    support_z: [B,S,4,D]
    support_y: [B,S]
    returns:   [B,S,D]
    """
    B, S, K, D = support_z.shape
    idx = support_y[..., None, None].expand(B, S, 1, D)
    chosen = support_z.gather(2, idx).squeeze(2)
    unchosen_mean = (support_z.sum(dim=2) - chosen) / float(K - 1)
    return chosen - unchosen_mean


def support_context(support_z: torch.Tensor) -> torch.Tensor:
    """Choice-independent semantic context for each support comparison: [B,S,D]."""
    return support_z.mean(dim=2)


def query_context(query_z: torch.Tensor) -> torch.Tensor:
    """Choice-independent semantic context for each query comparison: [B,Q,D]."""
    return query_z.mean(dim=2)


def static_profile(
    support_z: torch.Tensor,
    support_y: torch.Tensor,
    support_mask: torch.Tensor,
) -> torch.Tensor:
    ev = chosen_preference_evidence(support_z, support_y)
    w = support_mask.float()[..., None]
    denom = w.sum(dim=1).clamp_min(1.0)
    return (ev * w).sum(dim=1) / denom  # [B,D]


def retrieval_profile(
    support_z: torch.Tensor,
    support_y: torch.Tensor,
    support_mask: torch.Tensor,
    query_z: torch.Tensor,
    k: int,
    temperature: float,
) -> torch.Tensor:
    """Query-conditioned top-k weighted preference evidence: [B,Q,D]."""
    ev = chosen_preference_evidence(support_z, support_y)  # B,S,D
    sctx = F.normalize(support_context(support_z), dim=-1)  # B,S,D
    qctx = F.normalize(query_context(query_z), dim=-1)      # B,Q,D

    sim = torch.einsum("bqd,bsd->bqs", qctx, sctx)
    sim = sim.masked_fill(~support_mask[:, None, :], -1e9)

    kk = min(int(k), sim.shape[-1])
    topv, topi = torch.topk(sim, k=kk, dim=-1)

    # Invalid padded supports, if k exceeds a user's valid support count, retain
    # -1e9 and therefore receive essentially zero softmax mass.
    weights = F.softmax(topv / float(temperature), dim=-1)

    B, Q, _ = topi.shape
    D = ev.shape[-1]
    ev_exp = ev[:, None, :, :].expand(B, Q, ev.shape[1], D)
    gather_idx = topi[..., None].expand(B, Q, kk, D)
    selected = ev_exp.gather(2, gather_idx)
    return (weights[..., None] * selected).sum(dim=2)


class HistoryConditionedScorer(nn.Module):
    """Strong but compact order-invariant history baseline.

    The same scorer is used for static-profile and retrieval modes; only the way
    the history vector h is constructed differs.
    """
    def __init__(self, embed_dim: int, hidden: int = HIDDEN, dropout: float = DROPOUT):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.base_scorer = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.history_correction = nn.Sequential(
            nn.Linear(embed_dim * 3, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        # Start from query-only behavior; let history earn its correction.
        nn.init.zeros_(self.history_correction[-1].weight)
        nn.init.zeros_(self.history_correction[-1].bias)

    def forward(self, query_z: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        query_z: [B,Q,4,D]
        h:       [B,D] or [B,Q,D]
        returns: [B,Q,4]
        """
        B, Q, K, D = query_z.shape
        base = self.base_scorer(query_z).squeeze(-1)

        if h.ndim == 2:
            h = h[:, None, :].expand(B, Q, D)
        h4 = h[:, :, None, :].expand(B, Q, K, D)
        feat = torch.cat([query_z, h4, query_z * h4], dim=-1)
        corr = self.history_correction(feat).squeeze(-1)
        return base + corr


# -----------------------------------------------------------------------------
# Training/evaluation of static + retrieval baselines
# -----------------------------------------------------------------------------
@dataclass
class TrainResult:
    model: HistoryConditionedScorer
    history: pd.DataFrame
    best_val: float
    best_epoch: int


def batch_logits(
    model: HistoryConditionedScorer,
    batch: Dict[str, torch.Tensor],
    mode: str,
    retrieval_k: Optional[int],
    retrieval_temperature: float,
) -> torch.Tensor:
    if mode == "static":
        h = static_profile(batch["support_z"], batch["support_y"], batch["support_mask"])
    elif mode == "retrieval":
        if retrieval_k is None:
            raise ValueError("retrieval_k required")
        h = retrieval_profile(
            batch["support_z"], batch["support_y"], batch["support_mask"],
            batch["query_z"], retrieval_k, retrieval_temperature,
        )
    else:
        raise ValueError(mode)
    return model(batch["query_z"], h)


@torch.no_grad()
def evaluate_history_model(
    model: HistoryConditionedScorer,
    episodes: Sequence[Any],
    store: Any,
    device: str,
    mode: str,
    retrieval_k: Optional[int],
    retrieval_temperature: float,
) -> pd.DataFrame:
    model.eval()
    rows: List[Dict[str, Any]] = []

    for ep in episodes:
        sz = store.get(ep.support_items)[None, ...]
        sy = torch.tensor(ep.support_choices[None, :], dtype=torch.long, device=device)
        sm = torch.ones((1, len(ep.support_items)), dtype=torch.bool, device=device)
        qz = store.get(ep.query_items)[None, ...]

        batch = {"support_z": sz, "support_y": sy, "support_mask": sm, "query_z": qz}
        logits = batch_logits(
            model, batch, mode=mode, retrieval_k=retrieval_k,
            retrieval_temperature=retrieval_temperature,
        ).squeeze(0)
        P = F.softmax(logits, dim=-1).cpu().numpy()

        rows.extend(prob_rows(ep, P, model_name=mode))

    return pd.DataFrame(rows)


def evaluate_val_nll(
    model: HistoryConditionedScorer,
    episodes: Sequence[Any],
    store: Any,
    device: str,
    mode: str,
    retrieval_k: Optional[int],
    retrieval_temperature: float,
) -> float:
    d = evaluate_history_model(
        model, episodes, store, device, mode, retrieval_k, retrieval_temperature
    )
    return float(d["nll"].mean())


def train_history_model(
    ats06,
    model: HistoryConditionedScorer,
    train_histories: Sequence[Any],
    val_episodes: Sequence[Any],
    store: Any,
    result_dir: Path,
    mode: str,
    train_seed: int,
    epochs: int,
    users_per_batch: int,
    lr: float,
    weight_decay: float,
    patience: int,
    min_delta: float,
    retrieval_k_choices: Tuple[int, ...],
    retrieval_temperature: float,
    debug: bool,
) -> TrainResult:
    result_dir.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    rng = np.random.default_rng(train_seed + (1000 if mode == "retrieval" else 500))

    n_epochs = min(3, epochs) if debug else epochs
    best_val = float("inf")
    best_epoch = 0
    best_state = None
    early_best = float("inf")
    bad_epochs = 0
    hist: List[Dict[str, Any]] = []

    for epoch in range(1, n_epochs + 1):
        model.train()
        order = np.arange(len(train_histories))
        rng.shuffle(order)
        losses = []

        for start in range(0, len(order), users_per_batch):
            ids = order[start:start + users_per_batch]
            users = [train_histories[i] for i in ids]
            batch = ats06.make_train_batch(users, store, rng)
            if batch is None:
                continue

            k = None
            if mode == "retrieval":
                k = int(rng.choice(retrieval_k_choices))

            logits = batch_logits(
                model, batch, mode=mode, retrieval_k=k,
                retrieval_temperature=retrieval_temperature,
            )
            y = batch["query_y"]
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            losses.append(float(loss.item()))

        # For retrieval training, use the middle k as a stable stopping monitor.
        monitor_k = None
        if mode == "retrieval":
            monitor_k = retrieval_k_choices[len(retrieval_k_choices) // 2]

        val_nll = evaluate_val_nll(
            model, val_episodes, store, ats06.DEVICE,
            mode=mode, retrieval_k=monitor_k,
            retrieval_temperature=retrieval_temperature,
        )
        row = {
            "epoch": epoch,
            "train_nll": float(np.mean(losses)),
            "val_nll": val_nll,
            "monitor_k": monitor_k,
        }
        hist.append(row)
        pd.DataFrame(hist).to_csv(result_dir / "training_history.csv", index=False)

        if val_nll < best_val:
            best_val = val_nll
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save({
                "epoch": epoch,
                "val_nll": val_nll,
                "state_dict": best_state,
                "mode": mode,
            }, result_dir / "best.pt")

        if val_nll < early_best - min_delta:
            early_best = val_nll
            bad_epochs = 0
        else:
            bad_epochs += 1

        print(
            f"[{mode:9s}] epoch {epoch:03d} | train={row['train_nll']:.4f} | "
            f"val={val_nll:.4f}" +
            (f" | monitor_k={monitor_k}" if monitor_k is not None else "")
        )

        if patience > 0 and bad_epochs >= patience:
            print(f"[{mode}] early stop at {epoch}; best epoch={best_epoch}, val={best_val:.6f}")
            break

    if best_state is None:
        raise RuntimeError(f"No checkpoint produced for {mode}")
    model.load_state_dict(best_state)
    return TrainResult(model=model, history=pd.DataFrame(hist), best_val=best_val, best_epoch=best_epoch)


# -----------------------------------------------------------------------------
# ATS06 probability evaluation, retaining full 4-way probabilities for fusion
# -----------------------------------------------------------------------------
@torch.no_grad()
def evaluate_ats_probs(
    ats06,
    model: Any,
    episodes: Sequence[Any],
    store: Any,
    n_permutations: int,
    mode: str,
    seed_base: int,
) -> pd.DataFrame:
    if mode not in {"learned", "reset"}:
        raise ValueError(mode)
    model.eval()
    rows: List[Dict[str, Any]] = []

    for ep in episodes:
        if mode == "reset":
            P = ats06.predict_episode_queries(
                model, ep, store, torch.zeros(ats06.G_DIM, device=ats06.DEVICE)
            ).cpu().numpy()
        else:
            acc = None
            for rep in range(n_permutations):
                rng = np.random.default_rng(
                    ats06.deterministic_user_seed(ep.user_id, salt=seed_base + rep)
                )
                perm = rng.permutation(len(ep.support_items))
                g, _ = ats06.infer_g_for_episode(model, ep, store, perm)
                p = ats06.predict_episode_queries(model, ep, store, g).cpu().numpy()
                if acc is None:
                    acc = np.zeros_like(p, dtype=np.float64)
                acc += p / float(n_permutations)
            P = acc

        rows.extend(prob_rows(ep, P, model_name=f"ats_{mode}"))

    return pd.DataFrame(rows)


def prob_rows(ep: Any, P: np.ndarray, model_name: str) -> List[Dict[str, Any]]:
    out = []
    Y = np.asarray(ep.query_choices, dtype=int)
    pred = P.argmax(axis=1)
    for j, y in enumerate(Y):
        onehot = np.zeros(P.shape[1], dtype=float)
        onehot[y] = 1.0
        row = {
            "user_id": ep.user_id,
            "language": ep.language,
            "query_index": int(j),
            "model": model_name,
            "true_choice": int(y),
            "correct": int(pred[j] == y),
            "nll": float(-math.log(max(EPS, P[j, y]))),
            "brier": float(np.square(P[j] - onehot).sum()),
            "p_true": float(P[j, y]),
        }
        for c in range(P.shape[1]):
            row[f"p{c}"] = float(P[j, c])
        out.append(row)
    return out


# -----------------------------------------------------------------------------
# Fusion + metrics
# -----------------------------------------------------------------------------
def aligned_probs(a: pd.DataFrame, b: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    keys = ["user_id", "query_index", "true_choice", "language"]
    pa = a[keys + [f"p{i}" for i in range(4)]].copy()
    pb = b[keys + [f"p{i}" for i in range(4)]].copy()
    m = pa.merge(pb, on=keys, suffixes=("_a", "_b"), validate="one_to_one")
    A = m[[f"p{i}_a" for i in range(4)]].to_numpy(float)
    B = m[[f"p{i}_b" for i in range(4)]].to_numpy(float)
    return m, A, B


def fuse_log_probs(
    ats_df: pd.DataFrame,
    ret_df: pd.DataFrame,
    lam: float,
    model_name: str,
) -> pd.DataFrame:
    m, A, B = aligned_probs(ats_df, ret_df)
    logp = (1.0 - lam) * np.log(np.clip(A, EPS, 1.0)) + lam * np.log(np.clip(B, EPS, 1.0))
    logp = logp - logp.max(axis=1, keepdims=True)
    P = np.exp(logp)
    P = P / P.sum(axis=1, keepdims=True)

    rows = []
    for i, r in m.iterrows():
        y = int(r["true_choice"])
        onehot = np.zeros(4, dtype=float)
        onehot[y] = 1.0
        row = {
            "user_id": r["user_id"],
            "language": r["language"],
            "query_index": int(r["query_index"]),
            "model": model_name,
            "true_choice": y,
            "correct": int(int(P[i].argmax()) == y),
            "nll": float(-math.log(max(EPS, P[i, y]))),
            "brier": float(np.square(P[i] - onehot).sum()),
            "p_true": float(P[i, y]),
        }
        for c in range(4):
            row[f"p{c}"] = float(P[i, c])
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, name: str) -> Dict[str, Any]:
    return {
        "model": name,
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
    n_bootstrap: int,
    seed_offset: int,
) -> Dict[str, Any]:
    xa = per_user_metric(a, metric)
    xb = per_user_metric(b, metric)
    common = xa.index.intersection(xb.index)
    d = (xa.loc[common] - xb.loc[common]).to_numpy(float)
    rng = np.random.default_rng(BOOTSTRAP_SEED + seed_offset)
    boots = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        ix = rng.integers(0, len(d), size=len(d))
        boots[i] = d[ix].mean()
    return {
        "model_a": name_a,
        "model_b": name_b,
        "metric": metric,
        "n_users": int(len(d)),
        "delta": float(d.mean()),
        "ci_low": float(np.quantile(boots, 0.025)),
        "ci_high": float(np.quantile(boots, 0.975)),
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    args = build_arg_parser().parse_args()
    device = resolve_device(args.device)
    train_seed = int(args.seed)
    seed_everything(train_seed)

    ats06_dir = resolve_repo_path(args.ats06_dir)
    result_dir = resolve_repo_path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    k_choices = parse_k_choices(args.retrieval_k)

    metadata_path = ats06_dir / "experiment_metadata.json"
    split_path = ats06_dir / "user_splits.csv"
    best_path = ats06_dir / "best.pt"
    for p in (metadata_path, split_path, best_path):
        if not p.exists():
            raise FileNotFoundError(f"Required ATS06 artifact missing: {p}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    episode_seed = int(metadata.get("seed", 0))
    if episode_seed != 0:
        print(f"WARNING: ATS06 metadata seed is {episode_seed}, not 0. Reconstructing that seed's episodes.")

    ats06_source = discover_ats06_source(args.ats06_source)
    ats06 = load_module_from_path(ats06_source)
    configure_ats06_module(ats06, metadata, device, episode_seed)

    print("=" * 88)
    print("ATS07 — STRONG BASELINE CHALLENGE")
    print("=" * 88)
    print("REPO_ROOT:     ", REPO_ROOT)
    print("ATS06 source:  ", ats06_source)
    print("ATS06 result:  ", ats06_dir)
    print("RESULT_ROOT:   ", result_dir)
    print("DEVICE:        ", device)
    if device.startswith("cuda"):
        print("GPU:           ", torch.cuda.get_device_name(torch.device(device)))
    print("train_seed:    ", train_seed)
    print("episode_seed:  ", episode_seed)
    print("retrieval k:   ", k_choices)
    print("temperature:   ", args.retrieval_temperature)

    # Reuse the exact compact CA cache and frozen embeddings from ATS06.
    df, items = ats06.build_or_load_compact_data()

    # Important: user IDs are numeric-looking strings.  ATS06 creates split_df
    # in memory with string IDs, but re-reading user_splits.csv without an
    # explicit dtype makes pandas infer int64.  ats06.make_histories() looks up
    # split_map with str(uid), so int keys would silently produce zero users.
    split_df = pd.read_csv(split_path, dtype={"user_id": str, "language": str, "split": str})
    split_df["user_id"] = split_df["user_id"].astype(str)
    split_df["split"] = split_df["split"].astype(str).str.strip()

    E, item_to_idx, emb_meta = ats06.build_or_load_embeddings(items)
    store = ats06.EmbeddingStore(E, device)
    histories = ats06.make_histories(df, item_to_idx, split_df)
    val_eps = ats06.make_fixed_episodes(histories["val"])
    test_eps = ats06.make_fixed_episodes(histories["test"])

    expected_split_counts = split_df["split"].value_counts().to_dict()
    observed_counts = {k: len(histories[k]) for k in ("train", "val", "test")}
    if any(observed_counts[k] == 0 for k in observed_counts):
        raise RuntimeError(
            "ATS07 reconstructed an empty user split. "
            f"CSV split counts={expected_split_counts}; reconstructed={observed_counts}. "
            "Check user_id dtype/canonicalization before training."
        )

    if args.debug:
        histories["train"] = histories["train"][:120]
        val_eps = val_eps[:80]
        test_eps = test_eps[:80]

    print("users train/val/test:", len(histories["train"]), len(val_eps), len(test_eps))
    print("fixed embed dim:      ", store.embed_dim)

    # Load frozen ATS06 best model.
    ats_model = ats06.CEARHumanStateModel(store.embed_dim).to(device)
    ats_ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
    ats_model.load_state_dict(ats_ckpt["state_dict"])
    ats_model.eval()
    for p in ats_model.parameters():
        p.requires_grad_(False)
    print("ATS06 best epoch:     ", ats_ckpt.get("epoch"))
    print("ATS06 best val NLL:   ", ats_ckpt.get("val_nll"))

    # Train static profile baseline.
    seed_everything(train_seed + 10)
    static_model = HistoryConditionedScorer(store.embed_dim).to(device)
    static_result = train_history_model(
        ats06, static_model, histories["train"], val_eps, store,
        result_dir / "static_profile", "static", train_seed + 10,
        args.epochs, args.users_per_batch, args.lr, args.weight_decay,
        args.patience, args.min_delta, k_choices, args.retrieval_temperature,
        args.debug,
    )

    # Train one retrieval model with k augmentation; choose k on validation afterward.
    seed_everything(train_seed + 20)
    retrieval_model = HistoryConditionedScorer(store.embed_dim).to(device)
    retrieval_result = train_history_model(
        ats06, retrieval_model, histories["train"], val_eps, store,
        result_dir / "semantic_retrieval", "retrieval", train_seed + 20,
        args.epochs, args.users_per_batch, args.lr, args.weight_decay,
        args.patience, args.min_delta, k_choices, args.retrieval_temperature,
        args.debug,
    )

    # Validation: ATS probabilities + k selection + fusion lambda selection.
    print("\nEvaluating validation baselines...")
    val_ats = evaluate_ats_probs(
        ats06, ats_model, val_eps, store, args.val_ats_permutations,
        mode="learned", seed_base=episode_seed + 10000,
    )
    val_reset = evaluate_ats_probs(
        ats06, ats_model, val_eps, store, 1,
        mode="reset", seed_base=episode_seed + 20000,
    )
    val_static = evaluate_history_model(
        static_result.model, val_eps, store, device, "static", None,
        args.retrieval_temperature,
    )

    k_rows = []
    val_retrieval_by_k: Dict[int, pd.DataFrame] = {}
    for k in k_choices:
        d = evaluate_history_model(
            retrieval_result.model, val_eps, store, device, "retrieval", k,
            args.retrieval_temperature,
        )
        val_retrieval_by_k[k] = d
        k_rows.append({"k": k, "val_nll": float(d["nll"].mean())})
    k_df = pd.DataFrame(k_rows).sort_values("val_nll")
    k_df.to_csv(result_dir / "retrieval_k_selection.csv", index=False)
    best_k = int(k_df.iloc[0]["k"])
    val_retrieval = val_retrieval_by_k[best_k]

    lambda_rows = []
    best_lambda = None
    best_lambda_nll = float("inf")
    for lam in np.linspace(0.0, 1.0, 21):
        mix = fuse_log_probs(val_ats, val_retrieval, float(lam), "ats_plus_retrieval")
        nll = float(mix["nll"].mean())
        lambda_rows.append({"lambda_retrieval": float(lam), "val_nll": nll})
        if nll < best_lambda_nll:
            best_lambda_nll = nll
            best_lambda = float(lam)
    lambda_df = pd.DataFrame(lambda_rows)
    lambda_df.to_csv(result_dir / "fusion_lambda_selection.csv", index=False)

    print(f"selected retrieval k = {best_k}")
    print(f"selected fusion lambda_retrieval = {best_lambda:.2f}")

    # Test exactly once with validation-selected k/lambda.
    print("\nEvaluating TEST...")
    test_ats = evaluate_ats_probs(
        ats06, ats_model, test_eps, store, args.test_ats_permutations,
        mode="learned", seed_base=episode_seed + 10000,
    )
    test_reset = evaluate_ats_probs(
        ats06, ats_model, test_eps, store, 1,
        mode="reset", seed_base=episode_seed + 20000,
    )
    test_static = evaluate_history_model(
        static_result.model, test_eps, store, device, "static", None,
        args.retrieval_temperature,
    )
    test_retrieval = evaluate_history_model(
        retrieval_result.model, test_eps, store, device, "retrieval", best_k,
        args.retrieval_temperature,
    )
    test_fusion = fuse_log_probs(
        test_ats, test_retrieval, best_lambda, "ats_plus_retrieval"
    )

    named = {
        "ats_reset_query_only": test_reset,
        "static_profile": test_static,
        "semantic_retrieval": test_retrieval,
        "ats_g": test_ats,
        "ats_plus_retrieval": test_fusion,
    }

    for name, d in named.items():
        d.to_csv(result_dir / f"test_predictions_{name}.csv", index=False)

    summary = pd.DataFrame([summarize(d, name) for name, d in named.items()]).sort_values("nll")
    summary.to_csv(result_dir / "test_summary.csv", index=False)

    comparisons = [
        ("static_profile", "ats_reset_query_only"),
        ("semantic_retrieval", "ats_reset_query_only"),
        ("ats_g", "static_profile"),
        ("ats_g", "semantic_retrieval"),
        ("ats_plus_retrieval", "semantic_retrieval"),
        ("ats_plus_retrieval", "ats_g"),
    ]
    deltas = []
    seed_offset = 0
    for a, b in comparisons:
        for metric in ("accuracy", "nll", "brier"):
            deltas.append(bootstrap_delta(
                named[a], named[b], metric, a, b,
                n_bootstrap=args.bootstrap, seed_offset=seed_offset,
            ))
            seed_offset += 1
    delta_df = pd.DataFrame(deltas)
    delta_df.to_csv(result_dir / "bootstrap_deltas.csv", index=False)

    run_meta = {
        "ats06_dir": str(ats06_dir),
        "ats06_source": str(ats06_source),
        "ats06_best_epoch": int(ats_ckpt.get("epoch", -1)),
        "ats06_best_val_nll": float(ats_ckpt.get("val_nll", float("nan"))),
        "split_source": str(split_path),
        "episode_seed": episode_seed,
        "train_seed": train_seed,
        "device": device,
        "embed_dim": int(store.embed_dim),
        "static_trainable_params": int(sum(p.numel() for p in static_model.parameters() if p.requires_grad)),
        "retrieval_trainable_params": int(sum(p.numel() for p in retrieval_model.parameters() if p.requires_grad)),
        "static_best_epoch": static_result.best_epoch,
        "static_best_val_nll": static_result.best_val,
        "retrieval_best_epoch": retrieval_result.best_epoch,
        "retrieval_best_val_nll_monitor": retrieval_result.best_val,
        "retrieval_k_candidates": list(k_choices),
        "selected_retrieval_k": best_k,
        "retrieval_temperature": float(args.retrieval_temperature),
        "selected_fusion_lambda_retrieval": best_lambda,
        "selected_fusion_val_nll": best_lambda_nll,
        "val_ats_permutations": int(args.val_ats_permutations),
        "test_ats_permutations": int(args.test_ats_permutations),
        "semantic_retrieval_is_full_llm_rag": False,
    }
    (result_dir / "experiment_metadata.json").write_text(
        json.dumps(run_meta, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 88)
    print("ATS07 — TEST SUMMARY")
    print("=" * 88)
    print(summary.to_string(index=False))
    print("\nSelected retrieval k:", best_k)
    print("Selected retrieval fusion lambda:", best_lambda)
    print("\nKey bootstrap comparisons: model_a - model_b")
    print(delta_df.to_string(index=False))
    print("\nWrote:", result_dir)


if __name__ == "__main__":
    main()
