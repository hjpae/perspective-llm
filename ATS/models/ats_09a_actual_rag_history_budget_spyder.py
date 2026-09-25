# -*- coding: utf-8 -*-
"""
ATS09a — Actual LLM + RAG history-budget baseline
Spyder / Windows local runner

Place at:
    ATS/models/ats_09a_actual_rag_history_budget_spyder.py

This runner intentionally leaves the original ATS09a experiment untouched.
It:
    1. loads .env inside Python
    2. defaults retrieval to CPU
    3. removes the need to pass CLI arguments manually
    4. replaces the fragile free-form JSON API call with
       OpenAI Responses API + Structured Outputs
    5. uses a stable result directory so interrupted runs can resume
       from api_cache.jsonl
"""

# %% CONFIG
from __future__ import annotations

import importlib.util
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
from dotenv import load_dotenv

# ---------------------------------------------------------------------
# Edit these in Spyder
# ---------------------------------------------------------------------

#MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")
MODEL = "gpt-5-mini"

# SentenceTransformer retrieval only; API inference is remote.
DEVICE = "cpu"

# Start with True. Once the 10-query run succeeds, switch to False.
SMOKE_TEST = False
SMOKE_QUERIES = 10

# Full ATS09a design: one balanced query per PersonaMem benchmark persona.
MAX_QUERIES = 392  # once per users: 196 / twice: 392

WORKERS = 4
SEED = 0

# Explicitly keep reasoning small for this constrained probability task.
# Set None if you want the model's default reasoning behavior instead.
REASONING_EFFORT: Optional[str] = "minimal"

# Includes hidden reasoning tokens + visible structured output.
MAX_OUTPUT_TOKENS = 1200

API_RETRIES = 6

RUN_LABEL = "spyder_local"

# None -> automatically locate latest valid ATS08c result.
ATS08C_DIR: Optional[str] = None


# %% REPOSITORY / ENVIRONMENT SETUP
def find_repo_root() -> Path:
    """
    Find perspective-llm root independently of Spyder's working directory.
    """

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
        "Could not locate the perspective-llm repository root.\n"
        "Put this file in perspective-llm/ATS/models/ "
        "or launch Spyder from the repository."
    )


ROOT = find_repo_root()

load_dotenv(
    ROOT / ".env",
    override=False,
)

if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError(
        "OPENAI_API_KEY was not found.\n\n"
        f"Create:\n  {ROOT / '.env'}\n\n"
        "with:\n  OPENAI_API_KEY=your_key_here"
    )


def latest_ats08c_dir() -> Path:
    base = (
        ROOT
        / "results"
        / "ATS"
        / "personamem_adaptation"
    )

    if not base.exists():
        raise FileNotFoundError(
            f"ATS08c result directory not found:\n{base}"
        )

    candidates = sorted(
        [
            p
            for p in base.glob("seed0_*")
            if p.is_dir()
        ],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    for p in candidates:
        if (
            (p / "experiment_metadata.json").exists()
            and (p / "best.pt").exists()
        ):
            return p.resolve()

    raise FileNotFoundError(
        "Could not find a valid ATS08c run containing both "
        "experiment_metadata.json and best.pt."
    )


if ATS08C_DIR is None:
    ATS08C_PATH = latest_ats08c_dir()
else:
    p = Path(ATS08C_DIR)

    if not p.is_absolute():
        p = ROOT / p

    ATS08C_PATH = p.resolve()


EFFECTIVE_MAX_QUERIES = (
    SMOKE_QUERIES
    if SMOKE_TEST
    else MAX_QUERIES
)


def safe_name(x: str) -> str:
    return re.sub(
        r"[^A-Za-z0-9._-]+",
        "_",
        x,
    )


# Stable path = API cache survives reruns.
RESULT_DIR = (
    ROOT
    / "results"
    / "ATS"
    / "rag_history_budget"
    / (
        f"{RUN_LABEL}_"
        f"{safe_name(MODEL)}_"
        f"q{EFFECTIVE_MAX_QUERIES}_"
        f"seed{SEED}"
    )
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


print("=" * 88)
print("ATS09a — SPYDER LOCAL RUNNER")
print("=" * 88)

print("repo root:    ", ROOT)
print("ATS08c dir:   ", ATS08C_PATH)
print("model:        ", MODEL)
print("device:       ", DEVICE)
print("smoke test:   ", SMOKE_TEST)
print("max queries:  ", EFFECTIVE_MAX_QUERIES)
print("workers:      ", WORKERS)
print("reasoning:    ", REASONING_EFFORT)
print("result dir:   ", RESULT_DIR)


# %% LOAD ORIGINAL ATS09a
ORIGINAL_09A = (
    ROOT
    / "ATS"
    / "models"
    / "ats_09a_actual_rag_history_budget.py"
)

if not ORIGINAL_09A.exists():
    raise FileNotFoundError(
        f"Original ATS09a file not found:\n{ORIGINAL_09A}"
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
            f"Could not import {path}"
        )

    mod = importlib.util.module_from_spec(spec)

    sys.modules[name] = mod

    spec.loader.exec_module(mod)

    return mod


ats09 = load_module(
    ORIGINAL_09A,
    "ats09_original_spyder",
)


# %% ROBUST OPENAI API REPLACEMENT
PROBABILITY_SCHEMA = {
    "type": "object",
    "properties": {
        "A": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
        },
        "B": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
        },
        "C": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
        },
        "D": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
        },
    },
    "required": [
        "A",
        "B",
        "C",
        "D",
    ],
    "additionalProperties": False,
}


def extract_refusal(response) -> str:
    """
    Best-effort refusal extraction for Responses API.
    """

    try:
        output = getattr(
            response,
            "output",
            [],
        ) or []

        for item in output:

            if getattr(
                item,
                "type",
                None,
            ) != "message":
                continue

            content_items = getattr(
                item,
                "content",
                [],
            ) or []

            for content in content_items:

                if getattr(
                    content,
                    "type",
                    None,
                ) == "refusal":

                    return str(
                        getattr(
                            content,
                            "refusal",
                            "refusal",
                        )
                    )

    except Exception:
        pass

    return ""


def structured_api_call(
    client,
    model,
    dev,
    user,
    retries=API_RETRIES,
):
    """
    Drop-in replacement for ats09.api_call().

    Important:
    - Same dev/user prompt generated by original ATS09a.
    - Only the transport / output contract changes.
    - Structured Outputs removes the fragile manual JSON-generation step.
    """

    last_error = None

    for attempt in range(retries):

        try:

            kwargs = {
                "model": model,

                "input": [
                    {
                        "role": "developer",
                        "content": dev,
                    },
                    {
                        "role": "user",
                        "content": user,
                    },
                ],

                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": (
                            "preference_probabilities"
                        ),
                        "schema": (
                            PROBABILITY_SCHEMA
                        ),
                        "strict": True,
                    }
                },

                "max_output_tokens": (
                    MAX_OUTPUT_TOKENS
                ),
            }

            if REASONING_EFFORT is not None:

                kwargs["reasoning"] = {
                    "effort": (
                        REASONING_EFFORT
                    )
                }

            response = (
                client.responses.create(
                    **kwargs
                )
            )

            status = (
                getattr(
                    response,
                    "status",
                    "",
                )
                or ""
            )

            if (
                status
                and status != "completed"
            ):

                details = getattr(
                    response,
                    "incomplete_details",
                    None,
                )

                raise RuntimeError(
                    "Incomplete OpenAI response: "
                    f"status={status}, "
                    f"details={details}"
                )

            refusal = extract_refusal(
                response
            )

            if refusal:

                raise RuntimeError(
                    f"Model refusal: {refusal}"
                )

            raw = (
                getattr(
                    response,
                    "output_text",
                    "",
                )
                or ""
            ).strip()

            if not raw:

                raise RuntimeError(
                    "OpenAI response completed "
                    "without output_text."
                )

            obj = json.loads(raw)

            p = np.asarray(
                [
                    float(obj["A"]),
                    float(obj["B"]),
                    float(obj["C"]),
                    float(obj["D"]),
                ],
                dtype=float,
            )

            if (
                not np.all(np.isfinite(p))
                or np.any(p < 0)
                or p.sum() <= 0
            ):
                raise ValueError(
                    f"Invalid probabilities: {p}"
                )

            # Structured output asks for probabilities,
            # but normalize defensively against rounding.
            p = p / p.sum()

            usage = getattr(
                response,
                "usage",
                None,
            )

            output_details = getattr(
                usage,
                "output_tokens_details",
                None,
            )

            return {
                "probs": p.tolist(),

                "raw": raw,

                "response_id": str(
                    getattr(
                        response,
                        "id",
                        "",
                    )
                    or ""
                ),

                "input_tokens": int(
                    getattr(
                        usage,
                        "input_tokens",
                        0,
                    )
                    or 0
                ),

                "output_tokens": int(
                    getattr(
                        usage,
                        "output_tokens",
                        0,
                    )
                    or 0
                ),

                "reasoning_tokens": int(
                    getattr(
                        output_details,
                        "reasoning_tokens",
                        0,
                    )
                    or 0
                ),

                "total_tokens": int(
                    getattr(
                        usage,
                        "total_tokens",
                        0,
                    )
                    or 0
                ),
            }

        except Exception as e:

            last_error = e

            wait_seconds = min(
                2 ** attempt,
                16,
            )

            print(
                f"[API retry "
                f"{attempt + 1}/{retries}] "
                f"{type(e).__name__}: {e}"
            )

            if attempt + 1 < retries:

                print(
                    f"Retrying in "
                    f"{wait_seconds}s..."
                )

                time.sleep(
                    wait_seconds
                )

    raise RuntimeError(
        f"API failed after "
        f"{retries} attempts: "
        f"{last_error}"
    ) from last_error


# Replace only the API-call implementation.
ats09.api_call = structured_api_call


# %% RECORD LOCAL RUN CONFIG
runner_config = {
    "runner": (
        "ats_09a_actual_rag_history_budget_"
        "spyder.py"
    ),

    "original_experiment": str(
        ORIGINAL_09A
    ),

    "model": MODEL,

    "device": DEVICE,

    "smoke_test": SMOKE_TEST,

    "max_queries": (
        EFFECTIVE_MAX_QUERIES
    ),

    "workers": WORKERS,

    "seed": SEED,

    "reasoning_effort": (
        REASONING_EFFORT
    ),

    "max_output_tokens": (
        MAX_OUTPUT_TOKENS
    ),

    "ATS08c_dir": str(
        ATS08C_PATH
    ),

    "api_transport": (
        "OpenAI Responses API + "
        "strict Structured Outputs"
    ),
}


(
    RESULT_DIR
    / "spyder_runner_config.json"
).write_text(
    json.dumps(
        runner_config,
        indent=2,
    ),
    encoding="utf-8",
)


# %% BUILD ORIGINAL ATS09a ARGUMENTS
# main() inside the original file still uses argparse.
# Spyder does not need to deal with it: populate argv here.

sys.argv = [
    str(ORIGINAL_09A),

    "--ats08c-dir",
    str(ATS08C_PATH),

    "--personamem-root",
    str(
        ROOT
        / "data"
        / "personamem_v2"
    ),

    "--result-dir",
    str(RESULT_DIR),

    "--model",
    MODEL,

    "--device",
    DEVICE,

    "--max-queries",
    str(
        EFFECTIVE_MAX_QUERIES
    ),

    "--workers",
    str(WORKERS),

    "--seed",
    str(SEED),
]


# %% RUN ATS09a
if __name__ == "__main__":

    ats09.main()

    # Add local API details to the normal experiment metadata.
    metadata_path = (
        RESULT_DIR
        / "experiment_metadata.json"
    )

    if metadata_path.exists():

        metadata = json.loads(
            metadata_path.read_text(
                encoding="utf-8",
            )
        )

        metadata.update(
            {
                "local_runner": (
                    Path(__file__).name
                ),

                "api_transport": (
                    "OpenAI Responses API + "
                    "strict Structured Outputs"
                ),

                "reasoning_effort": (
                    REASONING_EFFORT
                ),

                "max_output_tokens": (
                    MAX_OUTPUT_TOKENS
                ),

                "retrieval_device": (
                    DEVICE
                ),
            }
        )

        metadata_path.write_text(
            json.dumps(
                metadata,
                indent=2,
            ),
            encoding="utf-8",
        )

    print(
        "\nFinished ATS09a local run."
    )

    print(
        "Results:",
        RESULT_DIR,
    )