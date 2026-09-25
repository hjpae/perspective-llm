# -*- coding: utf-8 -*-
"""
ATS09b — Actual LLM/RAG vs persistent ATS g
Spyder / Windows local analysis
=============================================

NO NEW OPENAI API CALLS.

Primary question
----------------
Does persistent person-state g add predictive information beyond explicit
retrieval of past preference memories?

Primary combined condition:
    delta_g(y) = log P_ATS(y | q, g) - log P_ATS(y | q, g=0)

    log P_RAG+g(y) = log P_RAG(y) + beta * delta_g(y)

Primary beta is fixed at 1.0. Other betas are sensitivity analyses only.

This is a READOUT-LEVEL complementary-signal test. It does not yet mean that
the LLM's internal representations are directly conditioned by g.
"""

# %% IMPORTS

from __future__ import annotations

import importlib.util
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


# %% CONFIG — EDIT HERE IN SPYDER

DEVICE = "cpu"
SEED = 0
N_BOOTSTRAP = 3000
EPS = 1e-8

# Primary, pre-specified g-correction strength.
PRIMARY_BETA = 1.0

# Sensitivity only.
# Do NOT select the best beta from benchmark performance.
BETA_SENSITIVITY = [0.25, 0.50, 1.00, 2.00]

# Match ATS08c benchmark evaluation.
ATS_PERMUTATIONS = 8

# Leave None for auto-discovery.
ATS09A_DIR: Optional[str] = None
ATS08C_DIR: Optional[str] = None

# Used only when ATS09A_DIR is None.
ATS09A_MODEL = "gpt-5-mini"
EXPECTED_QUERIES = 392

PERSONAMEM_ROOT = "data/personamem_v2"

ATS06_SOURCE = "ATS/models/ats_06_cear_g_ca.py"
ATS08B_SOURCE = "ATS/models/ats_08b_personamem_corrected_geometry.py"
ATS08C_SOURCE = "ATS/models/ats_08c_personamem_adaptation.py"

# Reuse locally cached ATS probabilities on reruns.
RECOMPUTE_ATS = False

# Make simple paper-analysis figures.
MAKE_FIGURES = True


# %% PATH HELPERS

def find_repo_root() -> Path:
    if "__file__" in globals():
        here = Path(__file__).resolve()

        for parent in [here.parent, *here.parents]:
            if (
                (parent / "ATS").exists()
                and (parent / "README.md").exists()
            ):
                return parent

    cwd = Path.cwd().resolve()

    for parent in [cwd, *cwd.parents]:
        if (
            (parent / "ATS").exists()
            and (parent / "README.md").exists()
        ):
            return parent

    raise RuntimeError(
        "Could not locate perspective-llm repository root.\n"
        "Put this file under perspective-llm/ATS/models/ "
        "or run Spyder from the repo."
    )


ROOT = find_repo_root()


def repo_path(value: str | Path) -> Path:
    p = Path(value).expanduser()

    return (
        p.resolve()
        if p.is_absolute()
        else (ROOT / p).resolve()
    )


def load_module(
    path: Path,
    name: str,
):
    spec = importlib.util.spec_from_file_location(
        name,
        str(path),
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Could not import module: {path}"
        )

    mod = importlib.util.module_from_spec(spec)

    sys.modules[name] = mod

    spec.loader.exec_module(mod)

    return mod


def read_json(path: Path) -> dict:
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def read_jsonl(path: Path) -> list[dict]:
    out = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            line = line.strip()

            if line:
                out.append(
                    json.loads(line)
                )

    return out


# %% AUTO-DISCOVERY


def discover_ats09a_dir() -> Path:

    if ATS09A_DIR is not None:

        p = repo_path(
            ATS09A_DIR
        )

        if not p.exists():
            raise FileNotFoundError(p)

        return p

    base = (
        ROOT
        / "results"
        / "ATS"
        / "rag_history_budget"
    )

    if not base.exists():
        raise FileNotFoundError(base)

    candidates = []

    for d in base.iterdir():

        if not d.is_dir():
            continue

        pred = (
            d
            / "predictions.csv"
        )

        mani = (
            d
            / "query_retrieval_manifest.jsonl"
        )

        meta = (
            d
            / "experiment_metadata.json"
        )

        if not (
            pred.exists()
            and mani.exists()
            and meta.exists()
        ):
            continue

        try:
            m = read_json(meta)

        except Exception:
            continue

        model = str(
            m.get(
                "model",
                "",
            )
        )

        nq = int(
            m.get(
                "selected_queries",
                m.get(
                    "selected_query_limit",
                    -1,
                ),
            )
        )

        if model != ATS09A_MODEL:
            continue

        if (
            EXPECTED_QUERIES > 0
            and nq != EXPECTED_QUERIES
        ):
            continue

        candidates.append(d)

    if not candidates:

        raise FileNotFoundError(
            "Could not auto-discover ATS09a "
            "result directory for\n"
            f"model={ATS09A_MODEL!r}, "
            f"queries={EXPECTED_QUERIES}.\n"
            "Set ATS09A_DIR explicitly in CONFIG."
        )

    return max(
        candidates,
        key=lambda p: p.stat().st_mtime,
    ).resolve()


def discover_ats08c_dir() -> Path:

    if ATS08C_DIR is not None:

        p = repo_path(
            ATS08C_DIR
        )

        if not p.exists():
            raise FileNotFoundError(p)

        return p

    base = (
        ROOT
        / "results"
        / "ATS"
        / "personamem_adaptation"
    )

    if not base.exists():
        raise FileNotFoundError(base)

    candidates = [
        p
        for p in base.glob(
            "seed0_*"
        )
        if (
            p.is_dir()
            and (
                p
                / "best.pt"
            ).exists()
            and (
                p
                / "experiment_metadata.json"
            ).exists()
        )
    ]

    if not candidates:

        raise FileNotFoundError(
            "No valid ATS08c directory "
            f"under {base}"
        )

    return max(
        candidates,
        key=lambda p: p.stat().st_mtime,
    ).resolve()


def resolve_ats06_dir_from_08c(
    ats08c_meta: dict,
) -> Path:
    """
    Repair the old /workspace path after
    moving the repo from Vast/Linux to Windows.
    """

    raw = ats08c_meta.get(
        "ats06_source_dir"
    )

    if raw:

        p = Path(raw)

        if p.exists():
            return p.resolve()

        # Example:
        # /workspace/.../rep_seed0_20260924_024922
        # ->
        # <local repo>/results/ATS/cear_g_ca/
        # rep_seed0_20260924_024922

        local = (
            ROOT
            / "results"
            / "ATS"
            / "cear_g_ca"
            / p.name
        )

        if local.exists():
            return local.resolve()

    # Fallback:
    # match by ATS06 best epoch used by ATS08c.

    base = (
        ROOT
        / "results"
        / "ATS"
        / "cear_g_ca"
    )

    target_epoch = (
        ats08c_meta.get(
            "initial_checkpoint_epoch"
        )
    )

    valid = []

    dirs = [base]

    if base.exists():

        dirs += [
            x
            for x in base.iterdir()
            if x.is_dir()
        ]

    for d in dirs:

        mp = (
            d
            / "experiment_metadata.json"
        )

        bp = (
            d
            / "best.pt"
        )

        if not (
            mp.exists()
            and bp.exists()
        ):
            continue

        try:
            m = read_json(mp)

        except Exception:
            continue

        if (
            target_epoch is None
            or int(
                m.get(
                    "best_epoch",
                    -999,
                )
            )
            == int(target_epoch)
        ):
            valid.append(d)

    if not valid:

        raise FileNotFoundError(
            "Could not resolve the ATS06 "
            "checkpoint used to initialize ATS08c."
        )

    return max(
        valid,
        key=lambda p: p.stat().st_mtime,
    ).resolve()


ATS09A_PATH = (
    discover_ats09a_dir()
)

ATS08C_PATH = (
    discover_ats08c_dir()
)

ATS08C_META = read_json(
    ATS08C_PATH
    / "experiment_metadata.json"
)

ATS06_DIR = (
    resolve_ats06_dir_from_08c(
        ATS08C_META
    )
)

ATS06_META = read_json(
    ATS06_DIR
    / "experiment_metadata.json"
)


OUT = (
    ROOT
    / "results"
    / "ATS"
    / "rag_g_comparison"
    / (
        f"{ATS09A_PATH.name}_"
        f"ats08c_{ATS08C_PATH.name}"
    )
).resolve()

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


print("=" * 96)
print(
    "ATS09b — ACTUAL LLM/RAG "
    "vs PERSISTENT g"
)
print("=" * 96)

print(
    "repo root:        ",
    ROOT,
)

print(
    "ATS09a source:    ",
    ATS09A_PATH,
)

print(
    "ATS08c source:    ",
    ATS08C_PATH,
)

print(
    "ATS06 init source:",
    ATS06_DIR,
)

print(
    "device:           ",
    DEVICE,
)

print(
    "ATS permutations: ",
    ATS_PERMUTATIONS,
)

print(
    "primary beta:     ",
    PRIMARY_BETA,
)

print(
    "output:           ",
    OUT,
)


# %% LOAD ATS09a RESULTS


P09 = pd.read_csv(
    ATS09A_PATH
    / "predictions.csv",
    dtype={
        "user_id": str,
        "query_id": str,
    },
)

MANIFEST = read_jsonl(
    ATS09A_PATH
    / "query_retrieval_manifest.jsonl"
)

META09 = read_json(
    ATS09A_PATH
    / "experiment_metadata.json"
)


P09["user_id"] = (
    P09["user_id"]
    .astype(str)
)

P09["query_id"] = (
    P09["query_id"]
    .astype(str)
)


PROB_COLS = [
    "p_A",
    "p_B",
    "p_C",
    "p_D",
]

KEYS = [
    "user_id",
    "query_id",
    "query_index",
    "true_choice",
]


missing = [
    c
    for c in PROB_COLS
    if c not in P09.columns
]

if missing:

    raise RuntimeError(
        "ATS09a predictions.csv lacks "
        f"probabilities: {missing}"
    )


query_keys = (
    P09[KEYS]
    .drop_duplicates()
    .reset_index(
        drop=True
    )
)


if len(MANIFEST) != len(query_keys):

    raise RuntimeError(
        "Manifest/prediction query mismatch: "
        f"{len(MANIFEST)} vs "
        f"{len(query_keys)}"
    )


seed09 = int(
    META09.get(
        "seed",
        SEED,
    )
)

seed08 = int(
    ATS08C_META.get(
        "train_seed",
        SEED,
    )
)

if (
    seed09 != SEED
    or seed08 != SEED
):

    raise RuntimeError(
        "Seed mismatch.\n"
        f"CONFIG SEED={SEED}, "
        f"ATS09a seed={seed09}, "
        f"ATS08c seed={seed08}"
    )


print("\n[ATS09a]")

print(
    "model:            ",
    META09.get(
        "model"
    ),
)

print(
    "selected queries: ",
    len(query_keys),
)

print(
    "users:            ",
    query_keys.user_id.nunique(),
)

print(
    "conditions:       ",
    sorted(
        P09.condition.unique()
    ),
)


# %% LOAD ATS MODULES + RECONSTRUCT BENCHMARK


A = load_module(
    repo_path(
        ATS06_SOURCE
    ),
    "ats06_09b",
)

B = load_module(
    repo_path(
        ATS08B_SOURCE
    ),
    "ats08b_09b",
)

C = load_module(
    repo_path(
        ATS08C_SOURCE
    ),
    "ats08c_09b",
)


# Configure ATS06 globals
# exactly as ATS08c did.

A.DEVICE = DEVICE

A.SEED = int(
    ATS06_META.get(
        "seed",
        0,
    )
)

A.G_DIM = int(
    ATS06_META[
        "g_dim"
    ]
)

A.COUPLER = str(
    ATS06_META[
        "coupler"
    ]
)

A.METRIC_RANK = int(
    ATS06_META[
        "metric_rank"
    ]
)

A.ALPHA_MIN = float(
    ATS06_META[
        "alpha_min"
    ]
)

A.ALPHA_MAX = float(
    ATS06_META[
        "alpha_max"
    ]
)

A.TEXT_ENCODER = str(
    ATS06_META[
        "text_encoder"
    ]
)


if hasattr(
    C,
    "seed_all",
):

    C.seed_all(
        SEED
    )


if hasattr(
    C,
    "TEST_PERMUTATIONS",
):

    if (
        int(C.TEST_PERMUTATIONS)
        != ATS_PERMUTATIONS
    ):

        print(
            "WARNING: ATS08c source "
            f"TEST_PERMUTATIONS="
            f"{C.TEST_PERMUTATIONS}, "
            f"but CONFIG uses "
            f"{ATS_PERMUTATIONS}."
        )


pmroot = repo_path(
    PERSONAMEM_ROOT
)


src = {}

all_items: Dict[
    str,
    dict,
] = {}


for split in [
    "train",
    "val",
    "benchmark",
]:

    df, items = C.prepare_csv(
        A,
        B,
        pmroot
        / "benchmark"
        / "text"
        / f"{split}.csv",
        split,
    )

    src[split] = df

    all_items.update(
        items
    )


E, item_to_idx = (
    C.build_embedding_cache(
        A,
        all_items,
        ATS06_META[
            "text_encoder"
        ],
        pmroot
        / "cache_ats08c",
        DEVICE,
        128,
    )
)


store = C.Store(
    E,
    DEVICE,
)


benchmark_histories = (
    C.make_histories(
        src[
            "benchmark"
        ],
        item_to_idx,
    )
)


benchmark_episodes = (
    C.fixed_episodes(
        benchmark_histories,
        SEED,
    )
)


ep_by_user = {
    str(ep.user_id): ep
    for ep in benchmark_episodes
}


idx_to_item_id = {
    int(v): k
    for k, v
    in item_to_idx.items()
}


print(
    "\n[ATS08c reconstruction]"
)

print(
    "benchmark episodes:",
    len(
        benchmark_episodes
    ),
)

print(
    "embedding dim:      ",
    store.embed_dim,
)

print(
    "g dim:              ",
    A.G_DIM,
)


# %% STRICT MANIFEST ALIGNMENT


for r in MANIFEST:

    uid = str(
        r[
            "user_id"
        ]
    )

    qj = int(
        r[
            "query_index"
        ]
    )

    if uid not in ep_by_user:

        raise RuntimeError(
            "Manifest user missing "
            "from ATS08c episodes: "
            f"{uid}"
        )

    ep = ep_by_user[
        uid
    ]

    if (
        qj < 0
        or qj
        >= len(
            ep.query_items
        )
    ):

        raise RuntimeError(
            "query_index out of range: "
            f"user={uid}, q={qj}"
        )

    y_ats = int(
        ep.query_choices[
            qj
        ]
    )

    y_manifest = int(
        r[
            "true_choice"
        ]
    )

    if (
        y_ats
        != y_manifest
    ):

        raise RuntimeError(
            "True-choice mismatch: "
            f"user={uid}, q={qj}, "
            f"ATS08c={y_ats}, "
            f"ATS09a={y_manifest}"
        )

    # Stronger identity check.

    if (
        "query_item" in r
        and isinstance(
            r[
                "query_item"
            ],
            dict,
        )
    ):

        manifest_item_id = (
            r[
                "query_item"
            ]
            .get(
                "item_id"
            )
        )

        local_item_id = (
            idx_to_item_id[
                int(
                    ep.query_items[
                        qj
                    ]
                )
            ]
        )

        if (
            manifest_item_id
            is not None
            and str(
                manifest_item_id
            )
            != str(
                local_item_id
            )
        ):

            raise RuntimeError(
                "ATS09a/ATS08c "
                "query-item mismatch:\n"
                f"user={uid}, q={qj}\n"
                f"manifest="
                f"{manifest_item_id}\n"
                f"local="
                f"{local_item_id}"
            )


print(
    "manifest alignment:  PASS"
)


# %% LOAD ADAPTED ATS08c CHECKPOINT


model = (
    A.CEARHumanStateModel(
        store.embed_dim
    )
    .to(
        DEVICE
    )
)


ck = torch.load(
    ATS08C_PATH
    / "best.pt",
    map_location="cpu",
    weights_only=False,
)


model.load_state_dict(
    ck[
        "state_dict"
    ]
)


model.eval()


for param in model.parameters():

    param.requires_grad_(
        False
    )


print(
    "ATS08c best epoch:  ",
    ck.get(
        "epoch",
        ATS08C_META.get(
            "best_epoch"
        ),
    ),
)


print(
    "ATS08c best val NLL:",
    ck.get(
        "val_nll",
        ATS08C_META.get(
            "best_val_nll"
        ),
    ),
)


# %% ATS FULL-PROBABILITY INFERENCE
# %% ON EXACT ATS09a QUERIES


ATS_CACHE = (
    OUT
    / "ats_selected_probabilities.csv"
)


def metric_row_from_prob(
    uid: str,
    query_id: str,
    query_index: int,
    y: int,
    condition: str,
    p: np.ndarray,
) -> dict:

    p = np.asarray(
        p,
        dtype=float,
    )

    p = (
        p
        / p.sum()
    )

    oh = np.zeros(
        4,
        dtype=float,
    )

    oh[y] = 1.0

    return {

        "user_id":
            str(uid),

        "query_id":
            str(query_id),

        "query_index":
            int(query_index),

        "true_choice":
            int(y),

        "condition":
            condition,

        "correct":
            int(
                int(
                    np.argmax(p)
                )
                == int(y)
            ),

        "p_true":
            float(
                p[y]
            ),

        "nll":
            float(
                -math.log(
                    max(
                        EPS,
                        p[y],
                    )
                )
            ),

        "brier":
            float(
                np.square(
                    p - oh
                ).sum()
            ),

        "p_A":
            float(
                p[0]
            ),

        "p_B":
            float(
                p[1]
            ),

        "p_C":
            float(
                p[2]
            ),

        "p_D":
            float(
                p[3]
            ),
    }


@torch.no_grad()
def compute_ats_selected_probs() -> pd.DataFrame:

    selected_by_user = (
        defaultdict(
            list
        )
    )

    for r in MANIFEST:

        selected_by_user[
            str(
                r[
                    "user_id"
                ]
            )
        ].append(r)

    rows = []

    users = sorted(
        selected_by_user
    )

    for ui, uid in enumerate(
        users,
        1,
    ):

        ep = ep_by_user[
            uid
        ]

        # -------------------------------------------------
        # g = 0
        # Same ATS architecture,
        # same query semantics,
        # no accumulated person-state.
        # -------------------------------------------------

        g0 = torch.zeros(
            A.G_DIM,
            device=store.E.device,
        )

        P0 = C.predict_queries(
            A,
            model,
            ep,
            store,
            g0,
        )

        # -------------------------------------------------
        # learned g
        # Average over ATS08c's support-order nuisance.
        # -------------------------------------------------

        Pg = np.zeros_like(
            P0,
            dtype=np.float64,
        )

        for rep in range(
            ATS_PERMUTATIONS
        ):

            rng = (
                np.random.default_rng(
                    C.det_seed(
                        uid,
                        SEED
                        + 10000,
                        salt=rep,
                    )
                )
            )

            perm = (
                rng.permutation(
                    len(
                        ep.support_items
                    )
                )
            )

            g = C.infer_g(
                A,
                model,
                ep,
                store,
                perm,
            )

            Pg += (
                C.predict_queries(
                    A,
                    model,
                    ep,
                    store,
                    g,
                )
                / float(
                    ATS_PERMUTATIONS
                )
            )

        # Only retain the exact
        # queries selected in ATS09a.

        for r in selected_by_user[
            uid
        ]:

            qj = int(
                r[
                    "query_index"
                ]
            )

            qid = str(
                r[
                    "query_id"
                ]
            )

            y = int(
                r[
                    "true_choice"
                ]
            )

            rows.append(
                metric_row_from_prob(
                    uid,
                    qid,
                    qj,
                    y,
                    "ats_reset",
                    P0[qj],
                )
            )

            rows.append(
                metric_row_from_prob(
                    uid,
                    qid,
                    qj,
                    y,
                    "ats_g",
                    Pg[qj],
                )
            )

        if (
            ui % 25 == 0
            or ui
            == len(users)
        ):

            print(
                "ATS probabilities: "
                f"processed "
                f"{ui}/"
                f"{len(users)} "
                "users"
            )

    return pd.DataFrame(
        rows
    )


if (
    ATS_CACHE.exists()
    and not RECOMPUTE_ATS
):

    ATS = pd.read_csv(
        ATS_CACHE,
        dtype={
            "user_id": str,
            "query_id": str,
        },
    )

    if (
        len(ATS)
        != 2
        * len(MANIFEST)
    ):

        print(
            "Cached ATS row count "
            "mismatch; recomputing."
        )

        ATS = (
            compute_ats_selected_probs()
        )

        ATS.to_csv(
            ATS_CACHE,
            index=False,
        )

    else:

        print(
            "\nLoading cached "
            "ATS probabilities:",
            ATS_CACHE,
        )

else:

    print(
        "\nComputing ATS g/reset "
        "probabilities on exact "
        "ATS09a queries..."
    )

    ATS = (
        compute_ats_selected_probs()
    )

    ATS.to_csv(
        ATS_CACHE,
        index=False,
    )


# %% STANDARDIZE ATS09a METRICS


LLM = (
    P09.copy()
    .rename(
        columns={
            "elicited_nll":
                "nll",

            "elicited_brier":
                "brier",
        }
    )
)


for col in [

    "input_tokens",
    "output_tokens",
    "reasoning_tokens",
    "total_tokens",

]:

    if col not in LLM.columns:

        LLM[col] = np.nan

    # ATS local compute is not
    # meaningfully comparable to
    # LLM API token count.

    ATS[col] = np.nan


def one_condition(
    df: pd.DataFrame,
    condition: str,
) -> pd.DataFrame:

    d = df[
        df.condition
        == condition
    ].copy()

    if (
        len(d)
        != len(MANIFEST)
    ):

        raise RuntimeError(
            f"Condition "
            f"{condition!r}: "
            f"{len(d)} rows; "
            f"expected "
            f"{len(MANIFEST)}"
        )

    return d


def align_two(
    a: pd.DataFrame,
    b: pd.DataFrame,
    suffix_a="_a",
    suffix_b="_b",
) -> pd.DataFrame:

    m = (
        a[
            KEYS
            + PROB_COLS
        ]
        .merge(
            b[
                KEYS
                + PROB_COLS
            ],
            on=KEYS,
            how="inner",
            validate="one_to_one",
            suffixes=(
                suffix_a,
                suffix_b,
            ),
        )
    )

    if (
        len(m)
        != len(MANIFEST)
    ):

        raise RuntimeError(
            "Probability alignment: "
            f"{len(m)} rows; "
            f"expected "
            f"{len(MANIFEST)}"
        )

    return m


ATS_G = one_condition(
    ATS,
    "ats_g",
)

ATS_0 = one_condition(
    ATS,
    "ats_reset",
)


# %% ISOLATE STATE-INDUCED LOG-ODDS CORRECTION


def state_logodds_delta(
    ats_g: pd.DataFrame,
    ats_reset: pd.DataFrame,
) -> pd.DataFrame:

    m = align_two(
        ats_g,
        ats_reset,
        "_g",
        "_0",
    )

    Pg = m[
        [
            f"{c}_g"
            for c
            in PROB_COLS
        ]
    ].to_numpy(
        float
    )

    P0 = m[
        [
            f"{c}_0"
            for c
            in PROB_COLS
        ]
    ].to_numpy(
        float
    )

    delta = (
        np.log(
            np.clip(
                Pg,
                EPS,
                1.0,
            )
        )
        -
        np.log(
            np.clip(
                P0,
                EPS,
                1.0,
            )
        )
    )

    # Per-query additive constant
    # is irrelevant under softmax.
    # Center for stability.

    delta -= (
        delta.mean(
            axis=1,
            keepdims=True,
        )
    )

    out = m[
        KEYS
    ].copy()

    for j, letter in enumerate(
        [
            "A",
            "B",
            "C",
            "D",
        ]
    ):

        out[
            f"delta_{letter}"
        ] = delta[:, j]

    out[
        "delta_l2"
    ] = np.sqrt(
        np.square(
            delta
        ).sum(
            axis=1
        )
    )

    out[
        "delta_abs_mean"
    ] = (
        np.abs(
            delta
        )
        .mean(
            axis=1
        )
    )

    return out


GDELTA = state_logodds_delta(
    ATS_G,
    ATS_0,
)


GDELTA.to_csv(
    OUT
    / "g_logodds_correction.csv",
    index=False,
)


# %% APPLY g CORRECTION TO ACTUAL LLM/RAG


def apply_gdelta(
    llm_condition_df: pd.DataFrame,
    beta: float,
    condition_name: str,
) -> pd.DataFrame:

    extra = [

        "actual_k",
        "n_support",

        "input_tokens",
        "output_tokens",
        "reasoning_tokens",
        "total_tokens",

    ]

    use_cols = (
        KEYS
        + PROB_COLS
        + [
            c
            for c in extra
            if c
            in llm_condition_df.columns
        ]
    )

    m = (
        llm_condition_df[
            use_cols
        ]
        .merge(
            GDELTA,
            on=KEYS,
            how="inner",
            validate="one_to_one",
        )
    )

    if (
        len(m)
        != len(MANIFEST)
    ):

        raise RuntimeError(
            "g-delta fusion: "
            f"{len(m)} rows; "
            f"expected "
            f"{len(MANIFEST)}"
        )

    Pr = m[
        PROB_COLS
    ].to_numpy(
        float
    )

    delta = m[
        [
            "delta_A",
            "delta_B",
            "delta_C",
            "delta_D",
        ]
    ].to_numpy(
        float
    )

    logits = (
        np.log(
            np.clip(
                Pr,
                EPS,
                1.0,
            )
        )
        +
        float(beta)
        * delta
    )

    logits -= (
        logits.max(
            axis=1,
            keepdims=True,
        )
    )

    P = np.exp(
        logits
    )

    P /= (
        P.sum(
            axis=1,
            keepdims=True,
        )
    )

    rows = []

    for i, r in m.iterrows():

        y = int(
            r[
                "true_choice"
            ]
        )

        oh = np.zeros(
            4,
            dtype=float,
        )

        oh[y] = 1.0

        p = P[i]

        row = {

            "user_id":
                str(
                    r[
                        "user_id"
                    ]
                ),

            "query_id":
                str(
                    r[
                        "query_id"
                    ]
                ),

            "query_index":
                int(
                    r[
                        "query_index"
                    ]
                ),

            "true_choice":
                y,

            "condition":
                condition_name,

            "beta_gdelta":
                float(beta),

            "correct":
                int(
                    int(
                        np.argmax(p)
                    )
                    == y
                ),

            "p_true":
                float(
                    p[y]
                ),

            "nll":
                float(
                    -math.log(
                        max(
                            EPS,
                            p[y],
                        )
                    )
                ),

            "brier":
                float(
                    np.square(
                        p - oh
                    ).sum()
                ),

            "p_A":
                float(
                    p[0]
                ),

            "p_B":
                float(
                    p[1]
                ),

            "p_C":
                float(
                    p[2]
                ),

            "p_D":
                float(
                    p[3]
                ),
        }

        for c in extra:

            if c in m.columns:

                row[c] = r[c]

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


RAG_CONDITIONS = [

    "query_only",

    "rag_1",

    "rag_3",

    "rag_5",

    "rag_10",

    "rag_all_support",

]


PRIMARY_COMBINED = {}


for cond in RAG_CONDITIONS:

    base = one_condition(
        LLM,
        cond,
    )

    PRIMARY_COMBINED[
        cond
    ] = apply_gdelta(

        base,

        PRIMARY_BETA,

        f"{cond}_plus_gdelta",
    )


# %% SUMMARY + USER-CLUSTER BOOTSTRAP


def summarize(
    df: pd.DataFrame,
    condition: str,
) -> dict:

    out = {

        "condition":
            condition,

        "n_query":
            int(
                len(df)
            ),

        "n_users":
            int(
                df.user_id.nunique()
            ),

        "accuracy":
            float(
                df.correct.mean()
            ),

        "nll":
            float(
                df.nll.mean()
            ),

        "brier":
            float(
                df.brier.mean()
            ),

        "mean_p_true":
            float(
                df.p_true.mean()
            ),

        "median_p_true":
            float(
                df.p_true.median()
            ),

        "p_true_lt_001":
            float(
                (
                    df.p_true
                    < 0.01
                ).mean()
            ),

        "p_true_lt_005":
            float(
                (
                    df.p_true
                    < 0.05
                ).mean()
            ),

        "median_nll":
            float(
                df.nll.median()
            ),

        "nll_p95":
            float(
                df.nll.quantile(
                    0.95
                )
            ),
    }

    out[
        "mean_actual_k"
    ] = (

        float(
            pd.to_numeric(
                df.actual_k,
                errors="coerce",
            ).mean()
        )

        if (
            "actual_k"
            in df.columns
        )

        else np.nan
    )

    out[
        "mean_input_tokens"
    ] = (

        float(
            pd.to_numeric(
                df.input_tokens,
                errors="coerce",
            ).mean()
        )

        if (
            "input_tokens"
            in df.columns
        )

        else np.nan
    )

    return out


def paired_user_delta(
    a: pd.DataFrame,
    b: pd.DataFrame,
    metric: str,
    name_a: str,
    name_b: str,
    seed_offset: int,
) -> dict:

    col = (
        "correct"
        if metric
        == "accuracy"
        else metric
    )

    aa = (
        a[
            [
                "user_id",
                "query_id",
                col,
            ]
        ]
        .rename(
            columns={
                col: "a"
            }
        )
    )

    bb = (
        b[
            [
                "user_id",
                "query_id",
                col,
            ]
        ]
        .rename(
            columns={
                col: "b"
            }
        )
    )

    m = aa.merge(

        bb,

        on=[
            "user_id",
            "query_id",
        ],

        how="inner",

        validate="one_to_one",
    )

    if (
        len(m)
        != len(MANIFEST)
    ):

        raise RuntimeError(
            "Paired comparison "
            f"{name_a} vs {name_b}: "
            f"{len(m)} rows; "
            f"expected "
            f"{len(MANIFEST)}"
        )

    m[
        "delta"
    ] = (
        m["a"]
        - m["b"]
    )

    # Cluster by user:
    # two queries/person do not count
    # as 392 independent people.

    u = (
        m.groupby(
            "user_id"
        )[
            "delta"
        ]
        .mean()
        .to_numpy(
            float
        )
    )

    rng = (
        np.random.default_rng(
            SEED
            + 50000
            + seed_offset
        )
    )

    idx = rng.integers(

        0,

        len(u),

        size=(
            N_BOOTSTRAP,
            len(u),
        ),
    )

    boots = (
        u[idx]
        .mean(
            axis=1
        )
    )

    return {

        "model_a":
            name_a,

        "model_b":
            name_b,

        "metric":
            metric,

        "n_users":
            int(
                len(u)
            ),

        "delta":
            float(
                u.mean()
            ),

        "ci_low":
            float(
                np.quantile(
                    boots,
                    0.025,
                )
            ),

        "ci_high":
            float(
                np.quantile(
                    boots,
                    0.975,
                )
            ),
    }


def bootstrap_condition_mean(
    df: pd.DataFrame,
    metric: str,
    seed_offset: int,
) -> tuple[
    float,
    float,
    float,
]:

    col = (
        "correct"
        if metric
        == "accuracy"
        else metric
    )

    u = (
        df.groupby(
            "user_id"
        )[
            col
        ]
        .mean()
        .to_numpy(
            float
        )
    )

    rng = (
        np.random.default_rng(
            SEED
            + 70000
            + seed_offset
        )
    )

    idx = rng.integers(

        0,

        len(u),

        size=(
            N_BOOTSTRAP,
            len(u),
        ),
    )

    boots = (
        u[idx]
        .mean(
            axis=1
        )
    )

    return (

        float(
            u.mean()
        ),

        float(
            np.quantile(
                boots,
                0.025,
            )
        ),

        float(
            np.quantile(
                boots,
                0.975,
            )
        ),
    )


# %% STACK ALL PRIMARY CONDITIONS


frames = []


for cond in RAG_CONDITIONS:

    d = one_condition(
        LLM,
        cond,
    ).copy()

    d[
        "family"
    ] = (
        "actual_llm_rag"
    )

    d[
        "beta_gdelta"
    ] = np.nan

    frames.append(
        d
    )


for cond in [

    "ats_reset",

    "ats_g",

]:

    d = one_condition(
        ATS,
        cond,
    ).copy()

    d[
        "family"
    ] = "ats"

    d[
        "beta_gdelta"
    ] = np.nan

    frames.append(
        d
    )


for _, d0 in (
    PRIMARY_COMBINED.items()
):

    d = d0.copy()

    d[
        "family"
    ] = (
        "actual_llm_rag_plus_gdelta"
    )

    frames.append(
        d
    )


ALL = pd.concat(

    frames,

    ignore_index=True,

    sort=False,
)


ALL.to_csv(
    OUT
    / "all_condition_predictions.csv",
    index=False,
)


SUMMARY = pd.DataFrame(

    [
        summarize(
            d,
            cond,
        )

        for cond, d
        in ALL.groupby(
            "condition",
            sort=False,
        )
    ]
)


SUMMARY.to_csv(
    OUT
    / "summary.csv",
    index=False,
)


# %% PRIMARY COMPARISONS


comparisons = []

seed_off = 0


# ---------------------------------------------------------
# A) ATS g standalone vs actual LLM/RAG
# ---------------------------------------------------------

for rag_cond in RAG_CONDITIONS:

    rag = one_condition(
        LLM,
        rag_cond,
    )

    for metric in [

        "accuracy",

        "nll",

        "brier",

    ]:

        comparisons.append(

            paired_user_delta(

                ATS_G,

                rag,

                metric,

                "ats_g",

                rag_cond,

                seed_off,
            )
        )

        seed_off += 1


# ---------------------------------------------------------
# B) CORE ATS09b TEST
#
# Does g add beyond the SAME RAG history budget?
# ---------------------------------------------------------

for rag_cond in RAG_CONDITIONS:

    rag = one_condition(
        LLM,
        rag_cond,
    )

    combo = (
        PRIMARY_COMBINED[
            rag_cond
        ]
    )

    combo_name = (
        f"{rag_cond}"
        "_plus_gdelta"
    )

    for metric in [

        "accuracy",

        "nll",

        "brier",

    ]:

        comparisons.append(

            paired_user_delta(

                combo,

                rag,

                metric,

                combo_name,

                rag_cond,

                seed_off,
            )
        )

        seed_off += 1


PRIMARY_DELTAS = pd.DataFrame(
    comparisons
)


PRIMARY_DELTAS.to_csv(
    OUT
    / "bootstrap_primary_deltas.csv",
    index=False,
)


# %% BETA SENSITIVITY
# %% EXPLORATORY ONLY — NO TEST-SET SELECTION


sensitivity_rows = []


for beta in BETA_SENSITIVITY:

    for rag_cond in RAG_CONDITIONS:

        d = apply_gdelta(

            one_condition(
                LLM,
                rag_cond,
            ),

            beta,

            (
                f"{rag_cond}"
                "_plus_gdelta_"
                f"beta{beta:g}"
            ),
        )

        s = summarize(
            d,
            d.condition.iloc[0],
        )

        s[
            "base_condition"
        ] = rag_cond

        s[
            "beta"
        ] = float(beta)

        sensitivity_rows.append(
            s
        )


BETA_DF = pd.DataFrame(
    sensitivity_rows
)


BETA_DF.to_csv(
    OUT
    / "beta_sensitivity_summary.csv",
    index=False,
)


# %% INCREMENTAL g EFFECT BY RAG HISTORY BUDGET


increment_rows = []


for rag_cond in RAG_CONDITIONS:

    combo_name = (
        f"{rag_cond}"
        "_plus_gdelta"
    )

    for metric in [

        "accuracy",

        "nll",

        "brier",

    ]:

        row = PRIMARY_DELTAS[

            (
                PRIMARY_DELTAS.model_a
                == combo_name
            )

            &

            (
                PRIMARY_DELTAS.model_b
                == rag_cond
            )

            &

            (
                PRIMARY_DELTAS.metric
                == metric
            )
        ]

        if len(row) != 1:

            raise RuntimeError(
                "Missing primary delta for "
                f"{combo_name} vs "
                f"{rag_cond}, "
                f"{metric}"
            )

        z = (
            row.iloc[0]
            .to_dict()
        )

        base = SUMMARY[
            SUMMARY.condition
            == rag_cond
        ].iloc[0]

        z[
            "rag_condition"
        ] = rag_cond

        z[
            "mean_actual_k"
        ] = base[
            "mean_actual_k"
        ]

        z[
            "mean_input_tokens"
        ] = base[
            "mean_input_tokens"
        ]

        increment_rows.append(
            z
        )


INCREMENT = pd.DataFrame(
    increment_rows
)


INCREMENT.to_csv(
    OUT
    / "g_increment_by_history_budget.csv",
    index=False,
)


# %% CONDITION-LEVEL BOOTSTRAP CIs


ci_rows = []

seed_off = 0


for cond, d in ALL.groupby(
    "condition",
    sort=False,
):

    for metric in [

        "accuracy",

        "nll",

        "brier",

    ]:

        mean, lo, hi = (
            bootstrap_condition_mean(
                d,
                metric,
                seed_off,
            )
        )

        seed_off += 1

        ci_rows.append(
            {
                "condition":
                    cond,

                "metric":
                    metric,

                "mean":
                    mean,

                "ci_low":
                    lo,

                "ci_high":
                    hi,
            }
        )


CI = pd.DataFrame(
    ci_rows
)


CI.to_csv(
    OUT
    / "condition_bootstrap_ci.csv",
    index=False,
)


# %% OPTIONAL FIGURES


BUDGET_ORDER = [

    "query_only",

    "rag_1",

    "rag_3",

    "rag_5",

    "rag_10",

    "rag_all_support",

]


def budget_x(
    cond: str,
) -> float:

    row = SUMMARY[
        SUMMARY.condition
        == cond
    ]

    if row.empty:
        return np.nan

    x = float(
        row.iloc[0][
            "mean_actual_k"
        ]
    )

    if np.isfinite(x):
        return x

    return (
        0.0
        if cond
        == "query_only"
        else np.nan
    )


def plot_metric(
    metric: str,
    ylabel: str,
    filename: str,
):

    fig, ax = plt.subplots(
        figsize=(
            7.4,
            4.8,
        )
    )

    for (
        family_label,
        suffix,
    ) in [

        (
            "Actual LLM/RAG",
            "",
        ),

        (
            "Actual LLM/RAG + g delta",
            "_plus_gdelta",
        ),

    ]:

        xs = []

        ys = []

        los = []

        his = []

        for base in BUDGET_ORDER:

            cond = (

                base

                if suffix == ""

                else (
                    f"{base}"
                    f"{suffix}"
                )
            )

            row = CI[

                (
                    CI.condition
                    == cond
                )

                &

                (
                    CI.metric
                    == metric
                )
            ]

            if row.empty:
                continue

            rec = row.iloc[0]

            xs.append(
                budget_x(
                    base
                )
            )

            ys.append(
                float(
                    rec[
                        "mean"
                    ]
                )
            )

            los.append(
                float(
                    rec[
                        "ci_low"
                    ]
                )
            )

            his.append(
                float(
                    rec[
                        "ci_high"
                    ]
                )
            )

        ys_arr = np.asarray(
            ys,
            dtype=float,
        )

        err = np.vstack(
            [
                ys_arr
                - np.asarray(
                    los,
                    dtype=float,
                ),

                np.asarray(
                    his,
                    dtype=float,
                )
                - ys_arr,
            ]
        )

        ax.errorbar(

            xs,

            ys,

            yerr=err,

            marker="o",

            capsize=3,

            label=family_label,
        )

    ax.set_xlabel(
        "Mean retrieved memories "
        "actually supplied to LLM"
    )

    ax.set_ylabel(
        ylabel
    )

    ax.set_title(
        "ATS09b: RAG history budget "
        "with/without persistent g correction"
    )

    ax.legend()

    ax.grid(
        alpha=0.2
    )

    fig.tight_layout()

    fig.savefig(
        OUT
        / filename,
        dpi=180,
    )

    plt.close(
        fig
    )


if MAKE_FIGURES:

    plot_metric(

        "accuracy",

        "Accuracy",

        "fig_accuracy_history_budget.png",
    )

    plot_metric(

        "nll",

        "NLL",

        "fig_nll_history_budget.png",
    )

    plot_metric(

        "brier",

        "Brier score",

        "fig_brier_history_budget.png",
    )


    # Direct incremental g effect
    # over matching RAG baseline.

    acc_inc = (
        INCREMENT[
            INCREMENT.metric
            == "accuracy"
        ]
        .sort_values(
            "mean_actual_k"
        )
    )

    fig, ax = plt.subplots(
        figsize=(
            7.4,
            4.8,
        )
    )

    x = (
        acc_inc
        .mean_actual_k
        .to_numpy(
            float
        )
    )

    y = (
        acc_inc
        .delta
        .to_numpy(
            float
        )
    )

    lo = (
        acc_inc
        .ci_low
        .to_numpy(
            float
        )
    )

    hi = (
        acc_inc
        .ci_high
        .to_numpy(
            float
        )
    )

    err = np.vstack(
        [
            y - lo,
            hi - y,
        ]
    )

    ax.errorbar(

        x,

        y,

        yerr=err,

        marker="o",

        capsize=3,
    )

    ax.axhline(
        0.0,
        linewidth=1,
    )

    ax.set_xlabel(
        "Mean retrieved memories "
        "actually supplied to LLM"
    )

    ax.set_ylabel(
        "Accuracy change: "
        "(RAG + g delta) - RAG"
    )

    ax.set_title(
        "Incremental contribution "
        "of persistent g beyond RAG"
    )

    ax.grid(
        alpha=0.2
    )

    fig.tight_layout()

    fig.savefig(
        OUT
        / "fig_g_increment_accuracy.png",
        dpi=180,
    )

    plt.close(
        fig
    )


# %% METADATA


metadata = {

    "experiment":
        (
            "ATS09b actual LLM/RAG vs "
            "PersonaMem-adapted persistent g"
        ),

    "ats09a_dir":
        str(
            ATS09A_PATH
        ),

    "ats09a_model":
        META09.get(
            "model"
        ),

    "selected_queries":
        int(
            len(MANIFEST)
        ),

    "selected_users":
        int(
            query_keys
            .user_id
            .nunique()
        ),

    "ats08c_dir":
        str(
            ATS08C_PATH
        ),

    "ats06_initialization_dir":
        str(
            ATS06_DIR
        ),

    "ats08c_best_epoch":
        ck.get(
            "epoch",
            ATS08C_META.get(
                "best_epoch"
            ),
        ),

    "ats_permutations":
        int(
            ATS_PERMUTATIONS
        ),

    "g_dim":
        int(
            A.G_DIM
        ),

    "primary_beta":
        float(
            PRIMARY_BETA
        ),

    "beta_sensitivity":
        [
            float(x)
            for x
            in BETA_SENSITIVITY
        ],

    "combination_rule":
        (
            "log P_combined = "
            "log P_actual_LLM_RAG + "
            "beta * "
            "(log P_ATS_g - "
            "log P_ATS_reset), "
            "then softmax"
        ),

    "primary_inference":
        (
            "paired user-cluster bootstrap "
            "of each RAG+g-delta condition "
            "against the matching RAG-only "
            "history budget"
        ),

    "scope_limit":
        (
            "readout-level complementary-"
            "signal test; not direct "
            "conditioning of LLM internal "
            "representations by g"
        ),

    "benchmark_tuning_policy":
        (
            "beta=1 fixed for primary "
            "analysis; sensitivity betas "
            "are not selected on benchmark; "
            "no single RAG k is selected "
            "on benchmark"
        ),
}


(
    OUT
    / "experiment_metadata.json"
).write_text(

    json.dumps(
        metadata,
        indent=2,
        ensure_ascii=False,
    ),

    encoding="utf-8",
)


# %% CONSOLE REPORT


print(
    "\n"
    + "=" * 96
)

print(
    "ATS09b — CONDITION SUMMARY"
)

print(
    "=" * 96
)


show_cols = [

    "condition",

    "n_query",

    "n_users",

    "accuracy",

    "nll",

    "brier",

    "mean_p_true",

    "mean_actual_k",

    "mean_input_tokens",

]


print(
    SUMMARY[
        show_cols
    ].to_string(
        index=False
    )
)


print(
    "\n"
    + "=" * 96
)

print(
    "CORE TEST — incremental g "
    "contribution beyond the SAME RAG budget"
)

print(
    "Positive accuracy delta is good; "
    "negative NLL/Brier delta is good."
)

print(
    "=" * 96
)


core = PRIMARY_DELTAS[

    PRIMARY_DELTAS
    .model_a
    .str.endswith(
        "_plus_gdelta"
    )

].copy()


print(
    core.to_string(
        index=False
    )
)


print(
    "\n"
    + "=" * 96
)

print(
    "ATS g standalone "
    "vs actual LLM/RAG"
)

print(
    "=" * 96
)


standalone = PRIMARY_DELTAS[

    PRIMARY_DELTAS.model_a
    == "ats_g"

].copy()


print(
    standalone.to_string(
        index=False
    )
)


print(
    "\n[g correction magnitude]"
)

print(
    "mean delta L2:      ",
    round(
        float(
            GDELTA
            .delta_l2
            .mean()
        ),
        4,
    ),
)

print(
    "median delta L2:    ",
    round(
        float(
            GDELTA
            .delta_l2
            .median()
        ),
        4,
    ),
)

print(
    "mean |delta|/class: ",
    round(
        float(
            GDELTA
            .delta_abs_mean
            .mean()
        ),
        4,
    ),
)


print(
    "\nWrote ATS09b outputs to:"
)

print(
    OUT
)


print(
    "\nPrimary files:"
)

print(
    "  summary.csv"
)

print(
    "  bootstrap_primary_deltas.csv"
)

print(
    "  g_increment_by_history_budget.csv"
)

print(
    "  beta_sensitivity_summary.csv"
)

print(
    "  all_condition_predictions.csv"
)

print(
    "  g_logodds_correction.csv"
)