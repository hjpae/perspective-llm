"""Prepare the ATS real-human corpora using the FULL Community Alignment corpus.

Why this version exists
-----------------------
The earlier EDA version used the first 5,000 streamed Community Alignment rows.
That is inappropriate for user-level longitudinal statistics because row order can
cluster annotators/waves and badly distort episodes-per-user distributions.

This version:
  * streams ALL Community Alignment rows from Hugging Face;
  * supports an authenticated Hugging Face token;
  * processes the corpus in chunks, so 90k+ conversations are never held in RAM;
  * preserves all candidate responses for preference supervision;
  * writes canonical CSV tables incrementally;
  * treats within-wave Community Alignment ordering as UNKNOWN;
  * keeps ThoughtTrace full (small enough to materialize at once).

Expected location:
    perspective-llm/ATS/data_pipeline/ats_02_prepare_real_corpora.py
"""

# %% Imports
from __future__ import annotations

import os
import shutil
from pathlib import Path
from itertools import islice

import pandas as pd
from datasets import load_dataset
from huggingface_hub import get_token

from real_corpora import (
    CANONICAL_TABLES,
    adapt_thoughttrace_rows,
    adapt_community_alignment_rows,
    adapt_talk2ai_local,
    write_tables,
)

# %% PROJECT PATHS
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]

OUT_ROOT = REPO_ROOT / "data" / "ATS" / "canonical"
RAW_ROOT = REPO_ROOT / "data" / "raw"

print("REPO_ROOT:", REPO_ROOT)
print("OUT_ROOT: ", OUT_ROOT)

# %% CONFIG
RUN_THOUGHTTRACE = True
RUN_COMMUNITY_ALIGNMENT = True
RUN_TALK2AI = False

# IMPORTANT: None means FULL corpus.
COMMUNITY_MAX_ROWS = None

# Stream the full corpus but adapt/write it in bounded chunks.
COMMUNITY_CHUNK_ROWS = 1000
PROGRESS_EVERY_ROWS = 5000

# Hugging Face auth:
# - easiest: run `hf auth login` once in a terminal;
# - alternatively set environment variable HF_TOKEN.
# Public datasets also work without auth, just at lower Hub rate limits.
HF_TOKEN = os.environ.get("HF_TOKEN") or get_token()

# We deliberately use CSV here because it supports safe incremental append.
OUTPUT_FORMAT = "csv"

# %% Helpers
def _append_df(df: pd.DataFrame, path: Path) -> None:
    if df is None or df.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(
        path,
        mode="a",
        header=not path.exists(),
        index=False,
    )


def _reset_source_dir(source: str) -> Path:
    out = OUT_ROOT / source
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    return out


def _iter_chunks(iterable, chunk_size: int, max_rows=None):
    iterator = iter(iterable)
    seen = 0
    while True:
        if max_rows is not None:
            remaining = max_rows - seen
            if remaining <= 0:
                return
            n = min(chunk_size, remaining)
        else:
            n = chunk_size

        chunk = list(islice(iterator, n))
        if not chunk:
            return

        seen += len(chunk)
        yield chunk, seen


def _concat_canonical_sources(sources):
    """Create combined canonical CSVs without loading all rows into memory."""
    combined = OUT_ROOT / "combined"
    if combined.exists():
        shutil.rmtree(combined)
    combined.mkdir(parents=True, exist_ok=True)

    for table in CANONICAL_TABLES:
        out_path = combined / f"{table}.csv"
        wrote = False

        for source in sources:
            src = OUT_ROOT / source / f"{table}.csv"
            if not src.exists():
                continue

            for chunk in pd.read_csv(src, chunksize=20000, keep_default_na=False):
                chunk.to_csv(
                    out_path,
                    mode="a",
                    header=not wrote,
                    index=False,
                )
                wrote = True

# %% ThoughtTrace
def prepare_thoughttrace():
    print("\n=== ThoughtTrace: FULL ===")
    out = _reset_source_dir("thoughttrace")

    ds = load_dataset(
        "SCAI-JHU/ThoughtTrace",
        split="train",
        streaming=True,
        token=HF_TOKEN,
    )

    rows = list(ds)
    tables = adapt_thoughttrace_rows(rows)
    write_tables(tables, out, fmt=OUTPUT_FORMAT)

    counts = {k: len(v) for k, v in tables.items()}
    print(counts)
    return counts

# %% Community Alignment
def prepare_community_alignment():
    print("\n=== Community Alignment: FULL STREAM ===")
    print("HF authentication:", "yes" if HF_TOKEN else "no (public/unauthenticated)")

    out = _reset_source_dir("community_alignment")

    ds = load_dataset(
        "facebook/community-alignment-dataset",
        split="train",
        streaming=True,
        token=HF_TOKEN,
    )

    # user_id is the only table requiring cross-chunk de-duplication.
    seen_users = set()

    totals = {name: 0 for name in CANONICAL_TABLES}
    source_rows = 0

    for batch, seen in _iter_chunks(
        ds,
        chunk_size=COMMUNITY_CHUNK_ROWS,
        max_rows=COMMUNITY_MAX_ROWS,
    ):
        tables = adapt_community_alignment_rows(batch)

        users = tables["users"]
        if not users.empty:
            keep = ~users["user_id"].astype(str).isin(seen_users)
            new_users = users.loc[keep].copy()
            seen_users.update(new_users["user_id"].astype(str).tolist())
            tables["users"] = new_users

        for name in CANONICAL_TABLES:
            df = tables.get(name, pd.DataFrame())
            _append_df(df, out / f"{name}.csv")
            totals[name] += len(df)

        source_rows = seen
        if source_rows % PROGRESS_EVERY_ROWS < COMMUNITY_CHUNK_ROWS:
            print(
                f"  streamed {source_rows:,} conversations | "
                f"users={len(seen_users):,} | "
                f"events={totals['events']:,} | "
                f"signals={totals['signals']:,}"
            )

    print("\nCommunity Alignment complete.")
    print("source conversations:", f"{source_rows:,}")
    print({
        "users": len(seen_users),
        "episodes": totals["episodes"],
        "events": totals["events"],
        "signals": totals["signals"],
    })

    return {
        "users": len(seen_users),
        "episodes": totals["episodes"],
        "events": totals["events"],
        "signals": totals["signals"],
    }

# %% Talk2AI
def prepare_talk2ai():
    print("\n=== Talk2AI ===")
    out = _reset_source_dir("talk2ai")
    tables = adapt_talk2ai_local(RAW_ROOT / "talk2ai")
    write_tables(tables, out, fmt=OUTPUT_FORMAT)
    counts = {k: len(v) for k, v in tables.items()}
    print(counts)
    return counts

# %% Main
def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    sources_written = []

    if RUN_THOUGHTTRACE:
        prepare_thoughttrace()
        sources_written.append("thoughttrace")

    if RUN_COMMUNITY_ALIGNMENT:
        prepare_community_alignment()
        sources_written.append("community_alignment")

    if RUN_TALK2AI:
        prepare_talk2ai()
        sources_written.append("talk2ai")

    print("\n=== Building combined canonical tables ===")
    _concat_canonical_sources(sources_written)
    print("Wrote:", (OUT_ROOT / "combined").resolve())

    print("\nNext: run ATS/eda/ats_03_real_corpora_eda.py")

if __name__ == "__main__":
    main()
