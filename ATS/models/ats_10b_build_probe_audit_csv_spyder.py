# %% ATS10a -> human audit CSV
"""
ATS10a probe JSONL -> Excel-friendly audit CSVs.

Purpose
-------
Convert the frozen 4-way human probe into a spreadsheet-friendly audit table
WITHOUT adding communication-strategy labels or changing the landscape.

Outputs
-------
1) probe_audit.csv
   - 1 row per scenario
   - A/B/C/D shown side-by-side for human inspection
   - audit fields are blank/default and intended to be edited manually
   - final_* fields are initialized to the original candidates

2) landscape_lookup.csv
   - 1 row per landscape candidate
   - intended ONLY as a lookup table when a probe candidate needs replacement
   - does not modify the original landscape JSONL
"""

# %%
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

# %% CONFIG
REPO_ROOT = Path(__file__).resolve().parents[2]

RESULT_REL = Path(
    "results/ATS/caregiving_comm_probe/"
    "gpt-5-mini_s48_pool40_seed20260927"
)

RESULT_DIR = REPO_ROOT / RESULT_REL
PROBE_JSONL = RESULT_DIR / "probe_items.jsonl"
LANDSCAPE_JSONL = RESULT_DIR / "landscape_items.jsonl"
AUDIT_CSV = RESULT_DIR / "probe_audit.csv"
LANDSCAPE_LOOKUP_CSV = RESULT_DIR / "landscape_lookup.csv"

EXPECTED_K = 4
SLOTS = ["A", "B", "C", "D"]

# %% HELPERS
def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON in {path.name}, line {lineno}: {e}"
                ) from e
    return rows

def validate_probe_item(item: dict) -> None:
    required = {
        "scenario_id",
        "context",
        "source_intent",
        "candidate_ids",
        "candidates",
    }
    missing = required - set(item)
    if missing:
        raise ValueError(
            f"{item.get('scenario_id', '<unknown>')}: missing fields {sorted(missing)}"
        )

    ids = item["candidate_ids"]
    texts = item["candidates"]

    if len(ids) != EXPECTED_K or len(texts) != EXPECTED_K:
        raise ValueError(
            f"{item['scenario_id']}: expected {EXPECTED_K} candidates, "
            f"got {len(ids)} ids and {len(texts)} texts."
        )

    if len(set(ids)) != EXPECTED_K:
        raise ValueError(f"{item['scenario_id']}: duplicate candidate IDs in probe.")

def safe_text(x) -> str:
    return "" if x is None else str(x)

# %% BUILD WIDE AUDIT TABLE
def build_probe_audit(probe_items: list[dict]) -> pd.DataFrame:
    out = []

    for item in probe_items:
        validate_probe_item(item)

        row = {
            "scenario_id": safe_text(item["scenario_id"]),
            "context": safe_text(item["context"]),
            "source_intent": safe_text(item["source_intent"]),
        }

        for slot, cid, text in zip(
            SLOTS, item["candidate_ids"], item["candidates"]
        ):
            row[f"{slot}_candidate_id"] = safe_text(cid)
            row[f"{slot}_candidate_text"] = safe_text(text)

            # Human audit fields
            row[f"{slot}_candidate_status"] = ""
            row[f"{slot}_reason_code"] = ""
            row[f"{slot}_auditor_note"] = ""

            # Downstream canonical candidate.
            # Keep original unless the auditor deliberately replaces it.
            row[f"{slot}_final_candidate_id"] = safe_text(cid)
            row[f"{slot}_final_candidate_text"] = safe_text(text)

        row["scenario_status"] = ""
        row["scenario_auditor_note"] = ""
        row["audit_complete"] = False
        out.append(row)

    df = pd.DataFrame(out)

    if df["scenario_id"].duplicated().any():
        dup = df.loc[df["scenario_id"].duplicated(), "scenario_id"].tolist()
        raise ValueError(f"Duplicate scenario IDs: {dup}")

    return df

# %% BUILD LANDSCAPE LOOKUP
def build_landscape_lookup(landscape_items: list[dict]) -> pd.DataFrame:
    out = []

    for item in landscape_items:
        sid = safe_text(item.get("scenario_id"))
        context = safe_text(item.get("context"))
        intent = safe_text(item.get("source_intent"))

        # Current multi-candidate JSONL structure
        if isinstance(item.get("candidate_ids"), list) and isinstance(
            item.get("candidates"), list
        ):
            ids = item["candidate_ids"]
            texts = item["candidates"]

            if len(ids) != len(texts):
                raise ValueError(
                    f"{sid}: landscape candidate_ids/candidates length mismatch."
                )

            for rank, (cid, text) in enumerate(zip(ids, texts), start=1):
                out.append(
                    {
                        "scenario_id": sid,
                        "context": context,
                        "source_intent": intent,
                        "landscape_rank": rank,
                        "candidate_id": safe_text(cid),
                        "candidate_text": safe_text(text),
                    }
                )
            continue

        # Tolerate one-candidate-per-record variants
        cid = item.get("candidate_id")
        text = item.get("candidate_text", item.get("candidate"))
        if cid is not None and text is not None:
            out.append(
                {
                    "scenario_id": sid,
                    "context": context,
                    "source_intent": intent,
                    "landscape_rank": item.get("landscape_rank", ""),
                    "candidate_id": safe_text(cid),
                    "candidate_text": safe_text(text),
                }
            )
            continue

        raise ValueError(
            f"{sid or '<unknown>'}: unrecognized landscape JSONL record structure."
        )

    df = pd.DataFrame(out)

    if not df.empty:
        dup = df.duplicated(subset=["scenario_id", "candidate_id"])
        if dup.any():
            bad = df.loc[dup, ["scenario_id", "candidate_id"]].head(10)
            raise ValueError(
                "Duplicate scenario/candidate IDs in landscape lookup:\n"
                + bad.to_string(index=False)
            )

    return df

# %% SAVE
def save_excel_friendly_csv(df: pd.DataFrame, path: Path) -> None:
    # utf-8-sig lets Excel on Windows open Unicode cleanly.
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")

# %% MAIN
def main():
    print("=" * 96)
    print("ATS10a — BUILD HUMAN AUDIT CSV")
    print("=" * 96)
    print(f"repo root:          {REPO_ROOT}")
    print(f"result dir:         {RESULT_DIR}")
    print(f"probe JSONL:        {PROBE_JSONL}")
    print(f"landscape JSONL:    {LANDSCAPE_JSONL}")
    print()

    probe_items = read_jsonl(PROBE_JSONL)
    landscape_items = read_jsonl(LANDSCAPE_JSONL)

    audit_df = build_probe_audit(probe_items)
    landscape_df = build_landscape_lookup(landscape_items)

    save_excel_friendly_csv(audit_df, AUDIT_CSV)
    save_excel_friendly_csv(landscape_df, LANDSCAPE_LOOKUP_CSV)

    print(f"probe scenarios:    {len(audit_df)}")
    print(f"probe candidates:   {len(audit_df) * EXPECTED_K}")
    print(f"landscape rows:     {len(landscape_df)}")
    print()
    print(f"audit CSV:          {AUDIT_CSV}")
    print(f"landscape lookup:   {LANDSCAPE_LOOKUP_CSV}")
    print()
    print("AUDIT RULES")
    print("  candidate_status: KEEP | REPLACE")
    print(
        "  reason_code: ROLE_DRIFT | CONTENT_ADDITION | CONTENT_OMISSION | "
        "INTENT_DRIFT | UNNATURAL_WORDING | OTHER"
    )
    print("  scenario_status: KEEP | REPLACE_ONE_OR_MORE | REBUILD_PROBE")
    print("  audit_complete: set TRUE only after the whole scenario has been reviewed.")
    print()
    print("IMPORTANT")
    print("  - Do not audit for strategy, warmth, directness, or diversity.")
    print("  - If replacing a candidate, choose another candidate from the SAME scenario")
    print("    in landscape_lookup.csv and overwrite that slot's final_candidate_id/text.")
    print("  - Do not alter the original *_candidate_id/text columns.")
    print("=" * 96)

# %% RUN
if __name__ == "__main__":
    main()
