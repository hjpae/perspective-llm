#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ATS10a — Recipient-conditioned caregiving communication probe builder
=====================================================================

Purpose
-------
Build a LOW-LABEL, Spyder-friendly communication probe for the next ATS transfer
experiment, now at a substantially larger sampling scale.

This script deliberately does NOT define or train on human-authored strategy labels
such as "autonomy", "validation", "directness", etc.  It also does NOT perform
LLM-based pragmatic clustering.  Instead it:

1) starts from a neutral caregiving communication context + source intent;
2) asks an LLM to sample many meaning-preserving realizations;
3) uses a separate LLM pass only as a semantic/factual QC heuristic;
4) removes only extremely near-duplicate items in frozen embedding space;
5) RETAINS THE FULL surviving local candidate cloud as the communication landscape;
6) selects only the 4-way human preference probe with farthest-point sampling;
7) saves scenario-centered embedding geometry for later latent-space analysis.

Natural convergence into a local semantic basin is not treated as an error.  The
scientific object is the within-intent structure itself; no pragmatic taxonomy is
inserted before that structure is observed.

The ONLY human target intended for ATS10b is the recipient's selected candidate.
Any QC fields in this builder are heuristics for keeping the stimulus set clean;
they are not communication-strategy labels and must not be used as ATS targets.

Expected repo location
----------------------
perspective-llm/ATS/models/ats_10a_caregiving_comm_probe_spyder.py

Spyder
------
Run the whole file with F5 / %runfile.  All settings are at the top of the file;
there are no required command-line arguments.

Dependencies
------------
pip install openai python-dotenv pydantic sentence-transformers pandas numpy
"""

# %% Imports
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer


# %% ---------------------------------------------------------------------------
# Spyder-friendly configuration
# -----------------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]

# Keep the model explicit here.  Do NOT read the model name from .env; this avoids
# stale Spyder-kernel environment variables silently changing experimental runs.
MODEL = "gpt-5-mini"
TEXT_ENCODER = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEVICE = "cpu"

SEED = 20260927

# Candidate construction.
# Full-scale default: 48 scenarios x roughly 40-48 accepted local realizations.
# We over-generate 48 at a time but retain the FULL post-QC candidate cloud.
GENERATE_PER_ROUND = 48
MAX_GENERATION_ROUNDS = 2
TARGET_POOL_K = 40
PROBE_K = 4

# Extremely conservative near-duplicate removal.  Same-intent candidates are
# EXPECTED to live in a tight semantic basin, so dedup is only cleanup for almost
# identical realizations.  It is not a diversity-enforcement step.
DEDUP_COSINE = 0.997

# Do not allow pathological long outputs to dominate embeddings / UI.
MAX_CANDIDATE_CHARS = 700

# API robustness.
API_RETRIES = 5
RETRY_BASE_SECONDS = 1.5

# First run can be shortened while debugging.
SMOKE_TEST = False
SMOKE_TEST_N_SCENARIOS = 2

# Existing per-scenario API cache is reused unless this is True.
FORCE_REGENERATE = False

SCENARIO_CSV = REPO_ROOT / "data" / "ATS" / "ats10a_caregiving_scenarios_full.csv"
RESULT_DIR = REPO_ROOT / "results" / "ATS" / "caregiving_comm_probe" / f"{MODEL}_s48_pool40_seed{SEED}"
CACHE_DIR = RESULT_DIR / "api_cache"

# Prompt versions are part of cache identity.  Increment when changing generation
# semantics, not for comments / cosmetic edits.
GEN_PROMPT_VERSION = "ats10a_gen_v3_scale"
QC_PROMPT_VERSION = "ats10a_qc_v1"


# %% Reproducibility
random.seed(SEED)
np.random.seed(SEED)


# %% Starter scenarios
# These are intentionally low-stakes communication/coordination situations rather
# than medical diagnosis, medication adherence, or safety-critical decision making.
# Edit the generated CSV directly if you want to replace/add scenarios.
STARTER_SCENARIOS = [
    {
        "scenario_id": 'cg_001',
        "context": "A family dinner that had been planned for Saturday has been moved to Sunday because the host's schedule changed.",
        "source_intent": 'Tell the older adult that the family dinner moved to Sunday and ask whether the new day works for them.',
    },
    {
        "scenario_id": 'cg_002',
        "context": 'The caregiver will arrive about one hour later than usual this afternoon because of an unavoidable delay.',
        "source_intent": 'Tell the older adult about the one-hour delay and coordinate the updated arrival time.',
    },
    {
        "scenario_id": 'cg_003',
        "context": 'The older adult has several grocery bags to bring inside. The caregiver is available to help but does not know whether help is wanted.',
        "source_intent": 'Offer help carrying the grocery bags and ask whether the older adult wants that help.',
    },
    {
        "scenario_id": 'cg_004',
        "context": 'A relative would like to have a video call later this week. The older adult has used video calling before but may or may not want help setting it up.',
        "source_intent": 'Tell the older adult about the proposed video call and ask whether they want help setting it up.',
    },
    {
        "scenario_id": 'cg_005',
        "context": 'A repair person needs access to the home to inspect the heater. Two possible arrival windows are available on the same day.',
        "source_intent": 'Explain that the heater inspection needs to be scheduled and ask which available arrival window the older adult prefers.',
    },
    {
        "scenario_id": 'cg_006',
        "context": "Two activities on the older adult's weekly calendar now overlap because one event changed time.",
        "source_intent": 'Point out the scheduling overlap and ask whether the older adult wants to review the two plans together.',
    },
    {
        "scenario_id": 'cg_007',
        "context": 'A shared ride the older adult expected is running late. There are several ordinary alternatives, including waiting or changing the pickup plan.',
        "source_intent": 'Tell the older adult that the ride is delayed and ask how they would like to handle the changed pickup plan.',
    },
    {
        "scenario_id": 'cg_008',
        "context": 'The caregiver would like to move their regular weekly phone call from the evening to the morning this week.',
        "source_intent": "Ask the older adult whether they would be comfortable moving this week's regular call to the morning.",
    },
    {
        "scenario_id": 'cg_009',
        "context": 'A kitchen repair will require access to the kitchen for about two hours. The exact visit time can be chosen from several ordinary time slots.',
        "source_intent": 'Tell the older adult about the two-hour kitchen repair visit and coordinate a time that works for them.',
    },
    {
        "scenario_id": 'cg_010',
        "context": 'A family member who had planned to visit this week can no longer come, but would like to arrange another day.',
        "source_intent": 'Tell the older adult that the visit cannot happen this week and ask whether they want to choose another day.',
    },
    {
        "scenario_id": 'cg_011',
        "context": 'A community center is holding an optional social event this weekend. The caregiver thinks the older adult might be interested but does not know.',
        "source_intent": 'Tell the older adult about the community event and ask whether they are interested in going.',
    },
    {
        "scenario_id": 'cg_012',
        "context": 'The caregiver found a simpler way to organize a recurring household task, but the current arrangement is still workable.',
        "source_intent": 'Suggest discussing the alternative way to organize the household task and ask whether the older adult wants to consider changing the current arrangement.',
    },
    {
        "scenario_id": 'cg_013',
        "context": 'A package expected today will now arrive tomorrow, and the delivery can be left at the door or handed over in person.',
        "source_intent": 'Tell the older adult that the package is delayed until tomorrow and ask how they would prefer to receive it.',
    },
    {
        "scenario_id": 'cg_014',
        "context": 'The gardener who usually comes on Tuesday needs to come on Wednesday this week instead.',
        "source_intent": 'Tell the older adult about the proposed change in gardening day and ask whether Wednesday works for them.',
    },
    {
        "scenario_id": 'cg_015',
        "context": 'A laundry machine repair visit needs to be arranged, and there are several ordinary appointment windows available.',
        "source_intent": 'Tell the older adult that the laundry machine repair needs to be scheduled and ask which available time works best for them.',
    },
    {
        "scenario_id": 'cg_016',
        "context": 'One item on the regular grocery list is unavailable at the store, and the caregiver can either choose a substitute or leave it off the order.',
        "source_intent": 'Tell the older adult that the grocery item is unavailable and ask whether they would prefer a substitute or to skip it this time.',
    },
    {
        "scenario_id": 'cg_017',
        "context": 'Several library books are due soon. They can be returned, or the caregiver can help check whether renewal is available.',
        "source_intent": 'Remind the older adult that the library books are due and ask whether they want help returning them or checking renewal.',
    },
    {
        "scenario_id": 'cg_018',
        "context": 'A relative has sent a new set of family photos digitally. The older adult can view them as they are or get help putting them somewhere easier to browse.',
        "source_intent": 'Tell the older adult about the new family photos and ask whether they want help viewing or organizing them.',
    },
    {
        "scenario_id": 'cg_019',
        "context": 'The family has started a group chat for everyday updates. Joining it is optional, and help with setup is available.',
        "source_intent": 'Tell the older adult about the family group chat and ask whether they want to join or get help setting it up.',
    },
    {
        "scenario_id": 'cg_020',
        "context": 'A person who usually joins the older adult for a weekly walk needs to move the walk to a different time this week.',
        "source_intent": 'Tell the older adult about the proposed change in walk time and ask whether the new time works for them.',
    },
    {
        "scenario_id": 'cg_021',
        "context": 'A community class the older adult has shown interest in is offered at two different times on the same day.',
        "source_intent": 'Tell the older adult about the two class times and ask which, if either, they would prefer.',
    },
    {
        "scenario_id": 'cg_022',
        "context": 'A neighbor has invited the older adult to have coffee later in the week. There is no obligation to accept.',
        "source_intent": "Tell the older adult about the neighbor's invitation and ask whether they are interested in going.",
    },
    {
        "scenario_id": 'cg_023',
        "context": 'A favorite chair needs to be moved temporarily while part of the room is cleaned, and there are several reasonable places it could go.',
        "source_intent": 'Explain that the chair needs to be moved temporarily and ask where the older adult would like it placed.',
    },
    {
        "scenario_id": 'cg_024',
        "context": 'Several ordinary household supplies are running low, and the caregiver is preparing a shopping list.',
        "source_intent": 'Tell the older adult which household supplies are running low and ask what they would like added to the shopping list.',
    },
    {
        "scenario_id": 'cg_025',
        "context": 'A meal delivery that was expected in the early afternoon will arrive later than planned.',
        "source_intent": 'Tell the older adult about the meal delivery delay and coordinate how they would like to handle the changed arrival time.',
    },
    {
        "scenario_id": 'cg_026',
        "context": 'The streaming service on the television has signed out. The older adult can leave it for later or get help signing back in.',
        "source_intent": 'Tell the older adult that the streaming service signed out and ask whether they want help signing back in.',
    },
    {
        "scenario_id": 'cg_027',
        "context": "The older adult's phone is nearly full and may stop saving new photos soon. The caregiver can help review files if wanted.",
        "source_intent": 'Tell the older adult that the phone storage is nearly full and ask whether they want help reviewing what can be cleared.',
    },
    {
        "scenario_id": 'cg_028',
        "context": 'A visitor who planned to arrive in the afternoon has asked whether coming earlier would be convenient.',
        "source_intent": 'Tell the older adult that the visitor asked about arriving earlier and ask whether that timing works for them.',
    },
    {
        "scenario_id": 'cg_029',
        "context": "A family holiday gathering will be held at a different relative's home than originally planned.",
        "source_intent": 'Tell the older adult about the change in gathering location and ask whether the new plan works for them.',
    },
    {
        "scenario_id": 'cg_030',
        "context": 'The regular grocery trip can happen either in the morning or later in the afternoon this week.',
        "source_intent": 'Tell the older adult about the two possible grocery-trip times and ask which they would prefer.',
    },
    {
        "scenario_id": 'cg_031',
        "context": 'A recurring household chore needs to happen on a different day this week because of a scheduling conflict.',
        "source_intent": 'Tell the older adult that the chore day needs to change this week and ask which alternative day works for them.',
    },
    {
        "scenario_id": 'cg_032',
        "context": 'A home cleaner who usually comes at the regular time has asked to come at a different time this week.',
        "source_intent": "Tell the older adult about the cleaner's proposed time change and ask whether the new time works for them.",
    },
    {
        "scenario_id": 'cg_033',
        "context": 'A furniture delivery can be scheduled in either of two available delivery windows.',
        "source_intent": 'Tell the older adult about the two furniture-delivery windows and ask which they prefer.',
    },
    {
        "scenario_id": 'cg_034',
        "context": 'The community center newsletter lists several optional activities taking place next week.',
        "source_intent": 'Tell the older adult about the upcoming community-center activities and ask whether any of them interest them.',
    },
    {
        "scenario_id": 'cg_035',
        "context": 'A friend wants to return an item they borrowed and can stop by at several ordinary times.',
        "source_intent": 'Tell the older adult that the friend wants to return the borrowed item and ask when they would like the visit to happen.',
    },
    {
        "scenario_id": 'cg_036',
        "context": "An ingredient planned for tonight's meal is unavailable, so the meal can either be adjusted or replaced with another familiar option.",
        "source_intent": 'Tell the older adult that the planned ingredient is unavailable and ask how they would like to adjust the meal.',
    },
    {
        "scenario_id": 'cg_037',
        "context": 'A family member has offered the older adult a ride for an ordinary errand, but taking the ride is optional.',
        "source_intent": 'Tell the older adult about the offered ride and ask whether they would like to use it.',
    },
    {
        "scenario_id": 'cg_038',
        "context": 'The household thermostat schedule could be changed so the home warms up earlier in the morning, but the current schedule still works.',
        "source_intent": 'Suggest the possible thermostat-schedule change and ask whether the older adult would like to try it.',
    },
    {
        "scenario_id": 'cg_039',
        "context": 'The shared calendar currently sends two reminders for the same recurring event, and one reminder could be removed.',
        "source_intent": 'Point out the duplicate calendar reminders and ask whether the older adult wants to keep both or remove one.',
    },
    {
        "scenario_id": 'cg_040',
        "context": 'A box of old family photos is available to sort into an album, but there is no need to do it now.',
        "source_intent": 'Tell the older adult that the photos could be organized into an album and ask whether they want to look through them together.',
    },
    {
        "scenario_id": 'cg_041',
        "context": 'A replacement television remote is available and can be set up now or left for another time.',
        "source_intent": 'Tell the older adult about the replacement remote and ask whether they want help setting it up.',
    },
    {
        "scenario_id": 'cg_042',
        "context": "The caregiver will be away for several days and wants to make a simple plan for watering the older adult's houseplants.",
        "source_intent": 'Tell the older adult about the need to arrange plant watering and ask how they would prefer to handle it.',
    },
    {
        "scenario_id": 'cg_043',
        "context": 'A delivery expected later this week requires someone to be present during a broad arrival window.',
        "source_intent": 'Tell the older adult that someone needs to be present for the delivery and ask which available arrangement works for them.',
    },
    {
        "scenario_id": 'cg_044',
        "context": 'The local library is hosting an optional talk next week on a topic the older adult may find interesting.',
        "source_intent": 'Tell the older adult about the library talk and ask whether they are interested in attending.',
    },
    {
        "scenario_id": 'cg_045',
        "context": 'A family video call needs to move from its usual time, and two alternative times are available.',
        "source_intent": 'Tell the older adult that the family video call time needs to change and ask which alternative they prefer.',
    },
    {
        "scenario_id": 'cg_046',
        "context": "The caregiver's regular visit this week needs to be shorter than usual because of another commitment.",
        "source_intent": "Tell the older adult that this week's visit needs to be shorter and ask whether the proposed timing still works for them.",
    },
    {
        "scenario_id": 'cg_047',
        "context": 'The caregiver has an idea for rearranging a kitchen storage area so frequently used items are grouped differently, but the current setup can remain.',
        "source_intent": 'Suggest discussing the alternative kitchen-storage arrangement and ask whether the older adult wants to try changing it.',
    },
    {
        "scenario_id": 'cg_048',
        "context": 'The caregiver is considering using a shared digital shopping list, while the current paper list remains available.',
        "source_intent": 'Tell the older adult about the shared digital shopping-list option and ask whether they want to try it or keep the current method.',
    },
]


# %% Structured API outputs
class CandidatePool(BaseModel):
    candidates: List[str] = Field(
        description="Distinct natural utterances that preserve the specified source intent and facts."
    )


class CandidateQC(BaseModel):
    candidate_id: str
    meaning_preserved: bool
    no_new_material_fact: bool
    no_core_intent_removed: bool
    natural_utterance: bool
    short_reason: str


class QCBatch(BaseModel):
    checks: List[CandidateQC]


# %% Utility helpers
def stable_hash(*parts: object) -> str:
    text = "|".join(str(x) for x in parts)
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


def normalize_space(x: object) -> str:
    s = "" if x is None else str(x)
    return re.sub(r"\s+", " ", s).strip()


def normalize_for_dedup(x: str) -> str:
    x = normalize_space(x).lower()
    x = re.sub(r"[^\w\s]", "", x)
    return re.sub(r"\s+", " ", x).strip()


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def extract_usage(response) -> Dict[str, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    inp = int(getattr(usage, "input_tokens", 0) or 0)
    out = int(getattr(usage, "output_tokens", 0) or 0)
    total = int(getattr(usage, "total_tokens", inp + out) or (inp + out))
    return {"input_tokens": inp, "output_tokens": out, "total_tokens": total}


def api_parse(client: OpenAI, *, model: str, system: str, user: str, schema):
    """Responses API + Structured Outputs with simple retry logic."""
    last = None
    for attempt in range(API_RETRIES):
        try:
            response = client.responses.parse(
                model=model,
                input=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                text_format=schema,
            )
            parsed = response.output_parsed
            if parsed is None:
                raise RuntimeError("Structured response returned no parsed object.")
            return parsed, extract_usage(response)
        except Exception as exc:
            last = exc
            if attempt + 1 >= API_RETRIES:
                break
            wait = min(RETRY_BASE_SECONDS * (2 ** attempt), 12.0)
            print(f"  API retry {attempt + 1}/{API_RETRIES - 1}: {type(exc).__name__}: {exc}")
            time.sleep(wait)
    raise RuntimeError(f"API call failed after {API_RETRIES} attempts: {last}")


# %% Scenario loading
def ensure_scenario_csv() -> pd.DataFrame:
    SCENARIO_CSV.parent.mkdir(parents=True, exist_ok=True)
    if not SCENARIO_CSV.exists():
        pd.DataFrame(STARTER_SCENARIOS).to_csv(SCENARIO_CSV, index=False)
        print(f"Created starter scenario file: {SCENARIO_CSV}")

    df = pd.read_csv(SCENARIO_CSV, dtype=str).fillna("")
    need = {"scenario_id", "context", "source_intent"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"Scenario CSV missing columns: {sorted(missing)}")

    for c in need:
        df[c] = df[c].map(normalize_space)

    df = df[(df.scenario_id != "") & (df.context != "") & (df.source_intent != "")].copy()
    if df.scenario_id.duplicated().any():
        dup = df.loc[df.scenario_id.duplicated(), "scenario_id"].tolist()
        raise ValueError(f"Duplicate scenario_id values: {dup}")

    if SMOKE_TEST:
        df = df.head(SMOKE_TEST_N_SCENARIOS).copy()
    return df.reset_index(drop=True)


# %% Candidate generation
def generation_system_prompt() -> str:
    return """You are constructing stimuli for a communication research probe.

Generate natural utterances from a caregiver to an older adult. The scientific goal is to sample a broad range of plausible communicative realizations WITHOUT assigning communication-strategy labels.

Hard constraints:
- Preserve the source intent and the material facts exactly.
- Do not add diagnoses, medical advice, safety claims, promises, threats, invented reasons, or new factual assumptions.
- Do not change who is deciding or what action is being requested/offered/communicated.
- Do not solve the situation on the person's behalf.
- Keep every candidate independently usable as the message that would actually be said.
- Sample natural variation in interpersonal framing, emphasis, wording, structure, and tone while preserving content.
- Do NOT artificially force extreme differences just to maximize diversity. Natural local convergence is acceptable.
- Do NOT name, annotate, or categorize the strategies you used.
- Do NOT include explanations, labels, headings, or numbering inside the candidate text.
- Prefer concise natural speech, usually one or two sentences.
"""


def generation_user_prompt(
    scenario_id: str,
    context: str,
    source_intent: str,
    n: int,
    existing: Optional[Sequence[str]] = None,
) -> str:
    existing = list(existing or [])
    parts = [
        f"Scenario ID: {scenario_id}",
        f"Context: {context}",
        f"Source intent: {source_intent}",
        "",
        f"Generate {n} candidate utterances that are not exact duplicates.",
        "Sample the natural range of ways this intent could be expressed while keeping the same intent and facts.",
        "Do not force artificial extremes merely to make the candidates look different.",
    ]
    if existing:
        parts += [
            "",
            "The following candidates already exist. Add further natural alternatives without repeating them exactly.",
            "Similarity is acceptable when it arises naturally; do not manufacture extreme contrasts.",
        ]
        for i, x in enumerate(existing, 1):
            parts.append(f"Existing {i}: {x}")
    return "\n".join(parts)


def generate_round(
    client: OpenAI,
    scenario: Dict[str, str],
    round_idx: int,
    existing: Sequence[str],
) -> Tuple[List[str], Dict[str, int]]:
    sid = scenario["scenario_id"]
    key = stable_hash(
        GEN_PROMPT_VERSION,
        MODEL,
        sid,
        scenario["context"],
        scenario["source_intent"],
        GENERATE_PER_ROUND,
        round_idx,
        *existing,
    )
    cache_path = CACHE_DIR / f"{sid}_gen_r{round_idx}_{key[:10]}.json"

    if cache_path.exists() and not FORCE_REGENERATE:
        obj = read_json(cache_path)
        return list(obj["candidates"]), dict(obj.get("usage", {}))

    parsed, usage = api_parse(
        client,
        model=MODEL,
        system=generation_system_prompt(),
        user=generation_user_prompt(
            sid,
            scenario["context"],
            scenario["source_intent"],
            GENERATE_PER_ROUND,
            existing=existing,
        ),
        schema=CandidatePool,
    )

    candidates = []
    seen = set()
    for x in parsed.candidates:
        x = normalize_space(x)
        if not x or len(x) > MAX_CANDIDATE_CHARS:
            continue
        k = normalize_for_dedup(x)
        if not k or k in seen:
            continue
        seen.add(k)
        candidates.append(x)

    write_json(
        cache_path,
        {
            "prompt_version": GEN_PROMPT_VERSION,
            "model": MODEL,
            "scenario_id": sid,
            "round": round_idx,
            "candidates": candidates,
            "usage": usage,
        },
    )
    return candidates, usage


# %% Semantic/factual QC heuristic
def qc_system_prompt() -> str:
    return """You are a conservative quality-control checker for communication research stimuli.

You are NOT labeling communication style or strategy. Do not classify tone, directness, autonomy, validation, empathy, or any other communication dimension.

For each candidate, judge only whether it is a clean alternative realization of the supplied source intent in the supplied context.

Set meaning_preserved=true only if the candidate still communicates the same practical intent.
Set no_new_material_fact=true only if it does not invent a material fact, reason, consequence, promise, diagnosis, recommendation, or assumption.
Set no_core_intent_removed=true only if it retains every essential request/offer/information component of the source intent.
Set natural_utterance=true only if it is a plausible message one person could naturally say to another.

Be strict about added facts and changed decision authority, but allow broad differences in interpersonal framing and wording. The framing differences are exactly what the later experiment is meant to discover.
"""


def qc_user_prompt(scenario: Dict[str, str], candidates: Sequence[str]) -> str:
    parts = [
        f"Scenario ID: {scenario['scenario_id']}",
        f"Context: {scenario['context']}",
        f"Source intent: {scenario['source_intent']}",
        "",
        "Candidates:",
    ]
    for i, c in enumerate(candidates):
        parts.append(f"c{i:03d}: {c}")
    parts.append("\nReturn one check for every candidate_id exactly once.")
    return "\n".join(parts)


def run_qc(
    client: OpenAI,
    scenario: Dict[str, str],
    candidates: Sequence[str],
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    sid = scenario["scenario_id"]
    key = stable_hash(QC_PROMPT_VERSION, MODEL, sid, *candidates)
    cache_path = CACHE_DIR / f"{sid}_qc_{key[:10]}.json"

    if cache_path.exists() and not FORCE_REGENERATE:
        obj = read_json(cache_path)
        return pd.DataFrame(obj["checks"]), dict(obj.get("usage", {}))

    parsed, usage = api_parse(
        client,
        model=MODEL,
        system=qc_system_prompt(),
        user=qc_user_prompt(scenario, candidates),
        schema=QCBatch,
    )

    rows = [x.model_dump() for x in parsed.checks]
    df = pd.DataFrame(rows)

    expected = {f"c{i:03d}" for i in range(len(candidates))}
    got = set(df.get("candidate_id", pd.Series(dtype=str)).astype(str))
    if expected != got:
        missing = sorted(expected - got)
        extra = sorted(got - expected)
        raise RuntimeError(f"QC candidate_id mismatch for {sid}: missing={missing}, extra={extra}")

    df = df.sort_values("candidate_id").reset_index(drop=True)
    write_json(
        cache_path,
        {
            "prompt_version": QC_PROMPT_VERSION,
            "model": MODEL,
            "scenario_id": sid,
            "checks": rows,
            "usage": usage,
        },
    )
    return df, usage


# %% Embedding + diversity selection
def cosine_matrix(E: np.ndarray) -> np.ndarray:
    E = np.asarray(E, dtype=np.float64)
    norm = np.linalg.norm(E, axis=1, keepdims=True)
    E = E / np.clip(norm, 1e-12, None)
    return E @ E.T


def greedy_remove_near_duplicates(
    texts: Sequence[str],
    E: np.ndarray,
    threshold: float,
) -> Tuple[List[int], List[int]]:
    """Keep first unique-enough item; return kept and removed indices."""
    sim = cosine_matrix(E)
    kept: List[int] = []
    removed: List[int] = []
    for i in range(len(texts)):
        if not kept:
            kept.append(i)
            continue
        if max(float(sim[i, j]) for j in kept) >= threshold:
            removed.append(i)
        else:
            kept.append(i)
    return kept, removed


def farthest_point_indices(E: np.ndarray, k: int) -> List[int]:
    """Deterministic cosine-space coverage sampling.

    Start near the normalized centroid, then repeatedly choose the item with the
    largest distance to its nearest selected item.
    """
    n = len(E)
    if k >= n:
        return list(range(n))
    if k <= 0:
        return []

    X = np.asarray(E, dtype=np.float64)
    X = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)
    centroid = X.mean(axis=0)
    centroid /= max(np.linalg.norm(centroid), 1e-12)
    first = int(np.argmax(X @ centroid))

    selected = [first]
    max_sim_to_selected = X @ X[first]

    while len(selected) < k:
        # cosine distance to nearest selected = 1 - max cosine similarity
        dist_to_nearest = 1.0 - max_sim_to_selected
        dist_to_nearest[selected] = -np.inf
        nxt = int(np.argmax(dist_to_nearest))
        selected.append(nxt)
        max_sim_to_selected = np.maximum(max_sim_to_selected, X @ X[nxt])

    return selected


def diversity_stats(E: np.ndarray) -> Dict[str, float]:
    if len(E) < 2:
        return {"n": int(len(E)), "pair_cos_mean": np.nan, "pair_cos_min": np.nan, "pair_cos_max": np.nan}
    S = cosine_matrix(E)
    vals = S[np.triu_indices(len(E), k=1)]
    return {
        "n": int(len(E)),
        "pair_cos_mean": float(vals.mean()),
        "pair_cos_min": float(vals.min()),
        "pair_cos_max": float(vals.max()),
    }


def scenario_centered_pca(rows: pd.DataFrame, embedding_matrix: np.ndarray) -> pd.DataFrame:
    """PCA after subtracting each scenario's candidate centroid.

    This reduces domination by scenario/topic content and preserves within-intent
    realization differences.  It is descriptive only; no strategy labels enter.
    """
    X = np.asarray(embedding_matrix, dtype=np.float64)
    Xc = X.copy()

    for sid, idx in rows.groupby("scenario_id", sort=False).groups.items():
        ii = np.asarray(list(idx), dtype=int)
        Xc[ii] -= Xc[ii].mean(axis=0, keepdims=True)

    # SVD PCA.  Coordinates are Xc @ V[:2].T.
    if len(Xc) < 2:
        pc = np.zeros((len(Xc), 2), dtype=np.float64)
        explained = [np.nan, np.nan]
    else:
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        ncomp = min(2, Vt.shape[0])
        pc0 = Xc @ Vt[:ncomp].T
        pc = np.zeros((len(Xc), 2), dtype=np.float64)
        pc[:, :ncomp] = pc0
        var = S ** 2
        denom = float(var.sum())
        explained0 = (var / denom)[:ncomp] if denom > 0 else np.full(ncomp, np.nan)
        explained = [np.nan, np.nan]
        for j in range(ncomp):
            explained[j] = float(explained0[j])

    out = rows.copy()
    out["centered_pc1"] = pc[:, 0]
    out["centered_pc2"] = pc[:, 1]
    out["pc1_explained_fraction_global"] = explained[0]
    out["pc2_explained_fraction_global"] = explained[1]
    return out


# %% Main builder
def main() -> None:
    load_dotenv(REPO_ROOT / ".env")
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            f"OPENAI_API_KEY not found. Put it in {REPO_ROOT / '.env'} or the environment."
        )

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 96)
    print("ATS10a — CAREGIVING COMMUNICATION PROBE BUILDER — SPYDER LOCAL")
    print("=" * 96)
    print("repo root:       ", REPO_ROOT)
    print("scenario file:   ", SCENARIO_CSV)
    print("result dir:      ", RESULT_DIR)
    print("model:           ", MODEL)
    print("encoder:         ", TEXT_ENCODER)
    print("device:          ", DEVICE)
    print("generate/round:  ", GENERATE_PER_ROUND)
    print("max rounds:      ", MAX_GENERATION_ROUNDS)
    print("target pool K:     ", TARGET_POOL_K)
    print("human probe K:   ", PROBE_K)
    print("strategy labels:  NONE (by design)")
    print()

    scenarios = ensure_scenario_csv()
    print(f"scenarios: {len(scenarios)}")

    client = OpenAI()
    print(f"Loading frozen encoder: {TEXT_ENCODER}")
    encoder = SentenceTransformer(TEXT_ENCODER, device=DEVICE)

    all_candidate_rows: List[Dict[str, object]] = []
    diversity_rows: List[Dict[str, object]] = []
    probe_records: List[Dict[str, object]] = []
    landscape_records: List[Dict[str, object]] = []
    token_totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    for sidx, srow in scenarios.iterrows():
        scenario = {
            "scenario_id": str(srow.scenario_id),
            "context": str(srow.context),
            "source_intent": str(srow.source_intent),
        }
        sid = scenario["scenario_id"]
        print("\n" + "-" * 96)
        print(f"[{sidx + 1}/{len(scenarios)}] {sid}")
        print("intent:", scenario["source_intent"])

        # 1) Over-generate.  Crucially, whether another round is needed is decided
        # AFTER semantic QC *and* embedding near-dedup.  In v1 this decision was
        # accidentally made before dedup, so a pool could look large enough (e.g.
        # 21 QC-pass items) and then collapse below PROBE_K after dedup.
        generated: List[str] = []
        generated_norm = set()
        qc_df_final: Optional[pd.DataFrame] = None
        texts_unique: List[str] = []
        E_unique: Optional[np.ndarray] = None
        near_dup_idx: List[int] = []

        for round_idx in range(1, MAX_GENERATION_ROUNDS + 1):
            new, usage = generate_round(client, scenario, round_idx, generated)
            for k in token_totals:
                token_totals[k] += int(usage.get(k, 0) or 0)

            for x in new:
                nx = normalize_for_dedup(x)
                if nx and nx not in generated_norm:
                    generated_norm.add(nx)
                    generated.append(x)

            print(f"  after generation round {round_idx}: {len(generated)} unique strings")

            # 2) LLM QC heuristic on the current accumulated pool.
            qc_df, usage_qc = run_qc(client, scenario, generated)
            for k in token_totals:
                token_totals[k] += int(usage_qc.get(k, 0) or 0)

            qc_df = qc_df.copy()
            qc_df["candidate_text"] = generated
            qc_df["qc_keep"] = (
                qc_df["meaning_preserved"].astype(bool)
                & qc_df["no_new_material_fact"].astype(bool)
                & qc_df["no_core_intent_removed"].astype(bool)
                & qc_df["natural_utterance"].astype(bool)
            )
            kept_qc = qc_df[qc_df.qc_keep].copy().reset_index(drop=True)
            print(f"  LLM QC keep: {len(kept_qc)}/{len(qc_df)}")

            qc_df_final = qc_df

            if len(kept_qc) == 0:
                texts_unique = []
                E_unique = np.zeros((0, 0), dtype=np.float32)
                near_dup_idx = []
            else:
                qc_texts = kept_qc.candidate_text.tolist()
                E_qc = encoder.encode(
                    qc_texts,
                    batch_size=64,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                ).astype(np.float32)
                uniq_idx, near_dup_idx = greedy_remove_near_duplicates(
                    qc_texts, E_qc, DEDUP_COSINE
                )
                texts_unique = [qc_texts[i] for i in uniq_idx]
                E_unique = E_qc[uniq_idx]

            print(
                f"  embedding dedup: {len(texts_unique)} kept / {len(kept_qc)} QC-pass "
                f"(removed {len(near_dup_idx)})"
            )

            # TARGET_POOL_K is a sampling-density target, not a diversity target.
            # Natural convergence is allowed; another round simply adds more draws
            # from the same-intent local neighborhood if QC leaves the pool small.
            if len(texts_unique) >= TARGET_POOL_K:
                break

            if round_idx < MAX_GENERATION_ROUNDS:
                print(
                    f"  post-dedup pool below sampling target ({len(texts_unique)}/{TARGET_POOL_K}); "
                    "drawing another round from the same intent..."
                )

        if qc_df_final is None or E_unique is None:
            raise RuntimeError(f"Internal error: no QC/embedding result for {sid}")

        if len(texts_unique) < PROBE_K:
            raise RuntimeError(
                f"{sid}: only {len(texts_unique)} candidates remain after all generation/QC/dedup rounds; "
                f"need at least {PROBE_K}. Inspect the candidate cache before changing thresholds."
            )

        if len(texts_unique) < TARGET_POOL_K:
            print(
                f"  NOTE: sampling target not reached after {MAX_GENERATION_ROUNDS} rounds; "
                f"retaining the full available pool K={len(texts_unique)}."
            )

        # Full local cloud = landscape.  No farthest-point subsampling is applied
        # to the discovery geometry; this preserves the observed basin density.
        texts_land = list(texts_unique)
        E_land = np.asarray(E_unique, dtype=np.float32)

        # Farthest-point sampling is used ONLY to make a compact 4-way human probe.
        probe_local_in_land = farthest_point_indices(E_land, min(PROBE_K, len(E_land)))
        probe_texts = [texts_land[i] for i in probe_local_in_land]

        # Stable content IDs, independent of A/B/C/D presentation position.
        candidate_ids = [stable_hash(sid, x)[:16] for x in texts_unique]
        land_candidate_ids = list(candidate_ids)
        probe_candidate_ids = [land_candidate_ids[i] for i in probe_local_in_land]

        # 4) Per-candidate audit table.
        qc_lookup = {
            normalize_for_dedup(r.candidate_text): r
            for r in qc_df_final.itertuples(index=False)
        }
        land_set = set(land_candidate_ids)
        probe_set = set(probe_candidate_ids)

        for i, (cid, text0) in enumerate(zip(candidate_ids, texts_unique)):
            q = qc_lookup.get(normalize_for_dedup(text0))
            all_candidate_rows.append({
                "scenario_id": sid,
                "context": scenario["context"],
                "source_intent": scenario["source_intent"],
                "candidate_id": cid,
                "candidate_text": text0,
                "meaning_preserved": bool(getattr(q, "meaning_preserved", True)),
                "no_new_material_fact": bool(getattr(q, "no_new_material_fact", True)),
                "no_core_intent_removed": bool(getattr(q, "no_core_intent_removed", True)),
                "natural_utterance": bool(getattr(q, "natural_utterance", True)),
                "qc_reason": str(getattr(q, "short_reason", "")),
                "landscape_selected": cid in land_set,
                "probe_selected": cid in probe_set,
            })

        # 5) Probe + landscape JSONL records.
        # Probe candidate order is deterministic but then shuffled by a scenario seed
        # so the first farthest-point item is not systematically "A".
        rng = np.random.default_rng(int(stable_hash(SEED, sid)[:8], 16))
        probe_order = np.arange(len(probe_texts))
        rng.shuffle(probe_order)
        probe_texts_shuf = [probe_texts[i] for i in probe_order]
        probe_ids_shuf = [probe_candidate_ids[i] for i in probe_order]

        probe_records.append({
            "scenario_id": sid,
            "context": scenario["context"],
            "source_intent": scenario["source_intent"],
            "candidate_ids": probe_ids_shuf,
            "candidates": probe_texts_shuf,
        })
        landscape_records.append({
            "scenario_id": sid,
            "context": scenario["context"],
            "source_intent": scenario["source_intent"],
            "candidate_ids": land_candidate_ids,
            "candidates": texts_land,
        })

        # 6) Diversity diagnostics; lower mean cosine = broader textual-semantic spread.
        for pool_name, pool_E in [
            ("landscape_full", E_land),
            ("probe", E_land[probe_local_in_land]),
        ]:
            d = diversity_stats(pool_E)
            diversity_rows.append({"scenario_id": sid, "pool": pool_name, **d})

        print(
            f"  final: unique={len(texts_unique)} | landscape={len(texts_land)} | probe={len(probe_texts_shuf)}"
        )

    # %% Save core tables
    cand_df = pd.DataFrame(all_candidate_rows)
    div_df = pd.DataFrame(diversity_rows)

    cand_path = RESULT_DIR / "candidate_audit.csv"
    div_path = RESULT_DIR / "diversity_summary.csv"
    cand_df.to_csv(cand_path, index=False)
    div_df.to_csv(div_path, index=False)

    with (RESULT_DIR / "probe_items.jsonl").open("w", encoding="utf-8") as f:
        for x in probe_records:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

    with (RESULT_DIR / "landscape_items.jsonl").open("w", encoding="utf-8") as f:
        for x in landscape_records:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

    # Human-choice template: NO semantic strategy fields, only the observed choice.
    choice_rows = []
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for rec in probe_records:
        row = {
            "recipient_id": "",
            "scenario_id": rec["scenario_id"],
            "choice": "",  # fill A/B/C/D (or 0/1/2/3 in ATS10b parser)
        }
        for i, text0 in enumerate(rec["candidates"]):
            row[f"candidate_{letters[i]}"] = text0
            row[f"candidate_{letters[i]}_id"] = rec["candidate_ids"][i]
        choice_rows.append(row)
    pd.DataFrame(choice_rows).to_csv(RESULT_DIR / "recipient_choices_template.csv", index=False)

    # %% Scenario-centered raw embedding geometry for the selected landscape.
    geometry_rows = []
    geometry_embeddings = []
    for rec in landscape_records:
        emb = encoder.encode(
            rec["candidates"],
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)
        for cid, text0, e in zip(rec["candidate_ids"], rec["candidates"], emb):
            geometry_rows.append({
                "scenario_id": rec["scenario_id"],
                "candidate_id": cid,
                "candidate_text": text0,
            })
            geometry_embeddings.append(e)

    geom_base = pd.DataFrame(geometry_rows).reset_index(drop=True)
    geom_E = np.asarray(geometry_embeddings, dtype=np.float32)
    geom_df = scenario_centered_pca(geom_base, geom_E)
    geom_df.to_csv(RESULT_DIR / "raw_centered_geometry.csv", index=False)
    np.save(RESULT_DIR / "landscape_embeddings.npy", geom_E.astype(np.float16))

    metadata = {
        "experiment": "ATS10a recipient-conditioned caregiving communication probe builder",
        "seed": SEED,
        "model": MODEL,
        "text_encoder": TEXT_ENCODER,
        "device": DEVICE,
        "scenario_csv": str(SCENARIO_CSV),
        "n_scenarios": int(len(scenarios)),
        "generate_per_round": GENERATE_PER_ROUND,
        "max_generation_rounds": MAX_GENERATION_ROUNDS,
        "target_pool_k": TARGET_POOL_K,
        "probe_k": PROBE_K,
        "dedup_cosine": DEDUP_COSINE,
        "generation_prompt_version": GEN_PROMPT_VERSION,
        "qc_prompt_version": QC_PROMPT_VERSION,
        "strategy_labels_used_for_generation_or_training": False,
        "qc_is_heuristic_not_target": True,
        "selection_method": "LLM semantic/factual QC -> ultra-conservative near-duplicate removal -> retain all surviving candidates for landscape; cosine farthest-point only for 4-way human probe",
        "generation_stop_rule": "continue sampling until post-QC, post-dedup pool reaches TARGET_POOL_K or MAX_GENERATION_ROUNDS",
        "landscape_subsampling": "none; full post-QC near-unique local candidate cloud retained",
        "pragmatic_clustering_used": False,
        "raw_geometry_centering": "subtract each scenario candidate centroid before global PCA",
        "n_landscape_candidates_total": int(sum(len(x["candidates"]) for x in landscape_records)),
        "mean_landscape_candidates_per_scenario": float(np.mean([len(x["candidates"]) for x in landscape_records])),
        "api_usage": token_totals,
    }
    write_json(RESULT_DIR / "experiment_metadata.json", metadata)

    print("\n" + "=" * 96)
    print("ATS10a COMPLETE")
    print("=" * 96)
    print("candidate audit:          ", cand_path)
    print("probe items:              ", RESULT_DIR / "probe_items.jsonl")
    print("landscape items:          ", RESULT_DIR / "landscape_items.jsonl")
    print("human choice template:    ", RESULT_DIR / "recipient_choices_template.csv")
    print("raw centered geometry:    ", RESULT_DIR / "raw_centered_geometry.csv")
    print("diversity summary:        ", div_path)
    print("metadata:                 ", RESULT_DIR / "experiment_metadata.json")
    print("API tokens:               ", token_totals)
    print()
    print("IMPORTANT: no communication-strategy labels or pragmatic clusters were used.")
    print("The full post-QC local candidate cloud is retained for landscape analysis.")
    print("Farthest-point sampling is used only for the compact 4-way human probe.")


# %% Run
if __name__ == "__main__":
    main()
