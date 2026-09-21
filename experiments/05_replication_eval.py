# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 23:33:33 2026

@author: Kardien
"""

# %% ============================================================
# 05_replication_eval.py
#
# Replication evaluation of the explicit slow-state architecture.
#
# Design:
#
#     SAME 200-step synthetic longitudinal scenario
#     × N independent LLM runs
#
# Purpose:
#
#     isolate stochastic variation in the LLM semantic parser
#     while keeping:
#
#         - scenario
#         - state architecture
#         - relation resolver
#         - hysteresis law
#         - thresholds
#
#     FIXED.
#
#
# Outputs:
#
#     per-run JSONL traces
#     per-run scalar metrics
#     replication summary
#     median + bootstrap 95% CI trajectories
#     fraction-active trajectory
#     semantic-error-rate trajectory
#
#
# IMPORTANT:
#
# This remains an engineering / eval-stage experiment.
# It does NOT establish scientific superiority over baselines.
# ================================================================


# %% Imports

from pathlib import Path
from copy import deepcopy

import os
import json
import random
import hashlib

from typing import Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field


# %% ============================================================
# Environment
# ================================================================

ROOT = Path(__file__).resolve().parents[1]

load_dotenv(
    ROOT / ".env"
)

MODEL = os.environ["OPENAI_MODEL"]

client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"]
)


# %% ============================================================
# Experiment configuration
# ================================================================

EXPERIMENT_VERSION = "05_replication_v1"

N_RUNS = 20

SCENARIO_SEED = 42

BOOTSTRAP_SAMPLES = 2000

BOOTSTRAP_SEED = 20260920


# ------------------------------------------------------------
# Set True ONLY if you intentionally want to delete all existing
# run logs and start again.
# ------------------------------------------------------------

RESET_ALL = False


# ------------------------------------------------------------
# Console reporting
# ------------------------------------------------------------

PRINT_EVERY = 50


# %% ============================================================
# Output structure
# ================================================================

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "05_replication_eval"
)

RUN_DIR = (
    OUTPUT_DIR
    / "runs"
)

RUN_DIR.mkdir(
    parents=True,
    exist_ok=True
)


METRICS_CSV = (
    OUTPUT_DIR
    / "replication_metrics.csv"
)


SUMMARY_JSON = (
    OUTPUT_DIR
    / "replication_summary.json"
)


# %% ============================================================
# State-dynamics parameters
#
# IMPORTANT:
# intentionally unchanged from 04.
# ================================================================

STATE_KEY = "morning_walk"

SCORE_DECAY = 0.92

ACTIVATION_THRESHOLD = 1.10

DEACTIVATION_THRESHOLD = 0.45

MONITORING_THRESHOLD = 0.25

MIN_EVIDENCE_FOR_ACTIVATION = 3

MAX_CHANGE_SCORE = 2.00

BASELINE_SUPPORT_STRENGTH = 0.18

PROVISIONAL_CONTRADICTION_BONUS = 0.12


# %% ============================================================
# Semantic orientation vocabulary
# ================================================================

TOWARD_OUTDOOR = "toward_outdoor_engagement"

AWAY_FROM_OUTDOOR = "away_from_outdoor_engagement"

NEUTRAL = "neutral"

UNCLEAR = "unclear"


# %% ============================================================
# LLM semantic schema
# ================================================================

class SemanticObservation(BaseModel):

    interpretation: str = Field(
        description=(
            "Concise description of what the utterance itself "
            "expresses, without comparison to persistent state."
        )
    )

    orientation: Literal[
        "toward_outdoor_engagement",
        "away_from_outdoor_engagement",
        "neutral",
        "unclear"
    ]

    temporal_scope: Literal[
        "single_event",
        "recent_pattern",
        "ongoing_preference",
        "future_intention",
        "unclear"
    ]

    semantic_strength: float = Field(
        ge=0.0,
        le=1.0
    )

    confidence: float = Field(
        ge=0.0,
        le=1.0
    )

    rationale: str


# %% ============================================================
# State-relative resolved interpretation
# ================================================================

class ResolvedInterpretation(BaseModel):

    relation_to_established_state: Literal[
        "supports",
        "contradicts",
        "ambiguous"
    ]

    relation_to_provisional_change: Literal[
        "supports",
        "contradicts",
        "no_active_change",
        "ambiguous"
    ]

    candidate_change_direction: Literal[
        "reduced_outdoor_engagement",
        "increased_outdoor_engagement",
        "none",
        "ambiguous"
    ]

    temporal_weight: float = Field(
        ge=0.0,
        le=1.0
    )

    evidence_for_longitudinal_change: float = Field(
        ge=0.0,
        le=1.0
    )


# %% ============================================================
# Initial state
# ================================================================

INITIAL_STATE = {

    STATE_KEY: {

        "established_state": {

            "value":
                (
                    "Usually takes and enjoys a morning walk "
                    "around 8 AM."
                ),

            "orientation":
                TOWARD_OUTDOOR,

            "confidence":
                0.95,

            "stability":
                0.95,

            "last_confirmed_step":
                0
        },


        "change_tracker": {

            "status":
                "inactive",

            "direction":
                AWAY_FROM_OUTDOOR,

            "score":
                0.0,

            "change_evidence_strength":
                0.0,

            "qualifying_evidence_count":
                0,

            "baseline_support_count":
                0,

            "first_signal_step":
                None,

            "last_signal_step":
                None,

            "activation_step":
                None,

            "last_deactivation_step":
                None,

            "peak_score":
                0.0,

            "provenance_ids":
                []
        },


        "provisional_change":
            None
    }
}


# %% ============================================================
# Synthetic utterance banks
#
# Same as 04.
# ================================================================

STABLE_BANK = [

    "I took my usual morning walk and enjoyed it.",

    "My morning walk felt good today.",

    "I went outside around 8 like usual.",

    "I enjoyed getting outside this morning.",

    "I kept my normal morning walk routine today.",

    "I was glad to get outside for my walk.",

    "Getting outside this morning felt refreshing.",

    "I took a walk before breakfast like usual.",

    "My usual morning walk went well today.",

    "I enjoyed being outside this morning."
]


TRANSIENT_AWAY_BANK = [

    "I'm tired today, so I skipped my morning walk.",

    "I don't feel like going outside today.",

    "I'm taking one morning off from walking today.",

    "I have a headache, so I skipped the walk this morning.",

    (
        "The weather is awful today, "
        "so I'm staying inside this morning."
    )
]


CHANGE_BANK = [

    "I've been skipping my morning walk a lot lately.",

    "I haven't really wanted to go outside lately.",

    "I'd rather stay inside in the mornings now.",

    "The morning walk doesn't appeal to me much anymore.",

    "I've continued skipping the morning walk.",

    "Staying indoors has felt preferable lately.",

    (
        "I don't really enjoy going outside "
        "in the morning anymore."
    ),

    (
        "I've been avoiding my usual "
        "morning walk recently."
    ),

    (
        "I keep choosing to stay inside "
        "instead of taking the walk."
    ),

    (
        "I don't think I want to keep doing "
        "the morning walk right now."
    ),

    (
        "My interest in going outside in the "
        "morning has really dropped."
    ),

    (
        "Lately I've preferred staying home "
        "instead of taking my walk."
    )
]


RECOVERY_BANK = [

    "I've started enjoying my morning walks again.",

    "I took my morning walk today and it felt good.",

    "I've been wanting to get outside again lately.",

    "Going outside in the morning feels normal again.",

    "I want to keep doing my morning walks.",

    "I've returned to my usual morning walk routine.",

    "I've been enjoying getting outside again.",

    (
        "My morning walks have been feeling "
        "good again lately."
    ),

    (
        "I've gone back to walking outside "
        "most mornings."
    ),

    (
        "I feel like my usual outdoor "
        "routine is returning."
    )
]


# %% ============================================================
# Scenario helpers
# ================================================================

def make_event(
    step: int,
    phase: str,
    utterance: str,
    expected_orientation: str
) -> dict:

    return {

        "step":
            step,

        "synthetic_day":
            step,

        "phase":
            phase,

        "utterance":
            utterance,

        "expected_orientation":
            expected_orientation
    }


# %% ============================================================
# Build fixed 200-step scenario
#
# IMPORTANT:
#
# Scenario is generated ONCE with a fixed seed and then reused
# identically across all N runs.
#
# Therefore the first replication isolates LLM stochasticity.
# ================================================================

def build_long_horizon_scenario(
    seed: int
) -> list[dict]:

    rng = random.Random(
        seed
    )

    events = []

    step = 1


    # ------------------------------------------------------------
    # Phase 1
    # t = 1–60
    # ------------------------------------------------------------

    for _ in range(60):

        events.append(

            make_event(

                step,

                "stable_baseline",

                rng.choice(
                    STABLE_BANK
                ),

                TOWARD_OUTDOOR
            )
        )

        step += 1


    # ------------------------------------------------------------
    # Phase 2
    # t = 61–65
    # ------------------------------------------------------------

    transient_pattern = [

        (
            rng.choice(
                TRANSIENT_AWAY_BANK
            ),
            AWAY_FROM_OUTDOOR
        ),

        (
            rng.choice(
                STABLE_BANK
            ),
            TOWARD_OUTDOOR
        ),

        (
            rng.choice(
                TRANSIENT_AWAY_BANK
            ),
            AWAY_FROM_OUTDOOR
        ),

        (
            rng.choice(
                STABLE_BANK
            ),
            TOWARD_OUTDOOR
        ),

        (
            rng.choice(
                TRANSIENT_AWAY_BANK
            ),
            AWAY_FROM_OUTDOOR
        )
    ]


    for utterance, orientation in transient_pattern:

        events.append(

            make_event(

                step,

                "transient_perturbation",

                utterance,

                orientation
            )
        )

        step += 1


    # ------------------------------------------------------------
    # Phase 3
    # t = 66–100
    # ------------------------------------------------------------

    for _ in range(35):

        events.append(

            make_event(

                step,

                "restabilization",

                rng.choice(
                    STABLE_BANK
                ),

                TOWARD_OUTDOOR
            )
        )

        step += 1


    # ------------------------------------------------------------
    # Phase 4
    # t = 101–145
    # ------------------------------------------------------------

    for _ in range(45):

        events.append(

            make_event(

                step,

                "genuine_change",

                rng.choice(
                    CHANGE_BANK
                ),

                AWAY_FROM_OUTDOOR
            )
        )

        step += 1


    # ------------------------------------------------------------
    # Phase 5
    # t = 146–160
    # ------------------------------------------------------------

    for i in range(15):

        if i % 2 == 0:

            utterance = rng.choice(
                CHANGE_BANK
            )

            orientation = (
                AWAY_FROM_OUTDOOR
            )

        else:

            utterance = rng.choice(
                RECOVERY_BANK
            )

            orientation = (
                TOWARD_OUTDOOR
            )


        events.append(

            make_event(

                step,

                "mixed_evidence",

                utterance,

                orientation
            )
        )

        step += 1


    # ------------------------------------------------------------
    # Phase 6
    # t = 161–200
    # ------------------------------------------------------------

    for _ in range(40):

        events.append(

            make_event(

                step,

                "recovery",

                rng.choice(
                    RECOVERY_BANK
                ),

                TOWARD_OUTDOOR
            )
        )

        step += 1


    assert len(events) == 200

    assert events[-1]["step"] == 200


    return events


SCENARIO = (
    build_long_horizon_scenario(
        SCENARIO_SEED
    )
)


# %% ============================================================
# Scenario fingerprint
#
# Prevents accidentally resuming old logs after scenario changes.
# ================================================================

SCENARIO_JSON = json.dumps(

    SCENARIO,

    sort_keys=True,

    ensure_ascii=False
)


SCENARIO_HASH = hashlib.sha256(

    SCENARIO_JSON.encode(
        "utf-8"
    )

).hexdigest()


# %% ============================================================
# Semantic parser
# ================================================================

def parse_semantic_observation(
    utterance: str
) -> tuple[SemanticObservation, dict]:

    system_prompt = """
You are a semantic observation extractor.

Analyze ONLY the current utterance.

Do NOT compare it with any prior user state.
Do NOT decide whether it supports or contradicts a persistent profile.
Do NOT decide whether a longitudinal change should be activated.

Extract only the utterance's own semantic content.

Orientation:

toward_outdoor_engagement:
    expresses going outside, walking, enjoyment of outdoor activity,
    return to outdoor routine, or desire for it.

away_from_outdoor_engagement:
    expresses avoiding, skipping, disliking, withdrawing from,
    or preferring not to engage in outdoor activity.

neutral:
    relevant but without meaningful directional preference.

unclear:
    direction cannot be determined.

Temporal scope:

single_event:
    one present or isolated event such as today or this morning.

recent_pattern:
    repeated or recent behavior such as lately, again,
    continued, recently, or repeatedly.

ongoing_preference:
    broader current preference or sustained orientation.

future_intention:
    explicit intention about future behavior.

unclear:
    temporal scope cannot be determined.

Return only the requested structured observation.
"""


    user_prompt = f"""
CURRENT UTTERANCE:

{utterance}
"""


    response = client.responses.parse(

        model=MODEL,

        reasoning={
            "effort": "minimal"
        },

        input=[

            {
                "role": "system",
                "content": system_prompt
            },

            {
                "role": "user",
                "content": user_prompt
            }
        ],

        text_format=SemanticObservation
    )


    observation = (
        response.output_parsed
    )


    if observation is None:

        raise RuntimeError(
            "Semantic parsing returned None."
        )


    usage = {}


    if response.usage is not None:

        usage = {

            "input_tokens":
                response.usage.input_tokens,

            "output_tokens":
                response.usage.output_tokens,

            "total_tokens":
                response.usage.total_tokens
        }


    return (
        observation,
        usage
    )


# %% ============================================================
# Temporal weighting
# ================================================================

def temporal_weight(
    temporal_scope: str
) -> float:

    weights = {

        "single_event":
            0.35,

        "recent_pattern":
            0.75,

        "ongoing_preference":
            0.90,

        "future_intention":
            1.00,

        "unclear":
            0.25
    }


    return weights[
        temporal_scope
    ]


# %% ============================================================
# Deterministic state-relative resolver
# ================================================================

def resolve_interpretation(
    state_before: dict,
    observation: SemanticObservation
) -> ResolvedInterpretation:

    domain = (
        state_before[
            STATE_KEY
        ]
    )


    established_orientation = (

        domain[
            "established_state"
        ][
            "orientation"
        ]
    )


    provisional = (

        domain[
            "provisional_change"
        ]
    )


    # ------------------------------------------------------------
    # Established relation
    # ------------------------------------------------------------

    if observation.orientation in [

        NEUTRAL,
        UNCLEAR

    ]:

        relation_to_established = (
            "ambiguous"
        )


    elif (

        observation.orientation
        == established_orientation

    ):

        relation_to_established = (
            "supports"
        )


    else:

        relation_to_established = (
            "contradicts"
        )


    # ------------------------------------------------------------
    # Provisional relation
    # ------------------------------------------------------------

    if provisional is None:

        relation_to_provisional = (
            "no_active_change"
        )


    else:

        provisional_orientation = (

            provisional[
                "orientation"
            ]
        )


        if observation.orientation in [

            NEUTRAL,
            UNCLEAR

        ]:

            relation_to_provisional = (
                "ambiguous"
        )


        elif (

            observation.orientation
            == provisional_orientation

        ):

            relation_to_provisional = (
                "supports"
        )


        else:

            relation_to_provisional = (
                "contradicts"
        )


    # ------------------------------------------------------------
    # Candidate direction
    # ------------------------------------------------------------

    if (

        observation.orientation
        == AWAY_FROM_OUTDOOR

    ):

        candidate_direction = (
            "reduced_outdoor_engagement"
        )


    elif (

        observation.orientation
        == TOWARD_OUTDOOR

    ):

        candidate_direction = (
            "increased_outdoor_engagement"
        )


    elif (

        observation.orientation
        == NEUTRAL

    ):

        candidate_direction = (
            "none"
        )


    else:

        candidate_direction = (
            "ambiguous"
        )


    # ------------------------------------------------------------
    # Longitudinal evidence
    # ------------------------------------------------------------

    t_weight = temporal_weight(
        observation.temporal_scope
    )


    if (

        relation_to_established
        == "contradicts"

    ):

        longitudinal_evidence = (

            observation.semantic_strength

            *

            observation.confidence

            *

            t_weight
        )


    else:

        longitudinal_evidence = 0.0


    longitudinal_evidence = max(

        0.0,

        min(
            1.0,
            longitudinal_evidence
        )
    )


    return ResolvedInterpretation(

        relation_to_established_state=
            relation_to_established,

        relation_to_provisional_change=
            relation_to_provisional,

        candidate_change_direction=
            candidate_direction,

        temporal_weight=
            t_weight,

        evidence_for_longitudinal_change=
            longitudinal_evidence
    )


# %% ============================================================
# Interface invariant checks
# ================================================================

def assert_interface_invariants(
    state_before: dict,
    observation: SemanticObservation,
    resolved: ResolvedInterpretation
):

    domain = (
        state_before[
            STATE_KEY
        ]
    )


    established_orientation = (

        domain[
            "established_state"
        ][
            "orientation"
        ]
    )


    if (

        observation.orientation
        == established_orientation

    ):

        assert (

            resolved.relation_to_established_state
            == "supports"
        )


    elif observation.orientation not in [

        NEUTRAL,
        UNCLEAR

    ]:

        assert (

            resolved.relation_to_established_state
            == "contradicts"
        )


    provisional = (

        domain[
            "provisional_change"
        ]
    )


    if provisional is None:

        assert (

            resolved.relation_to_provisional_change
            == "no_active_change"
        )


    else:

        provisional_orientation = (

            provisional[
                "orientation"
            ]
        )


        if (

            observation.orientation
            == provisional_orientation

        ):

            assert (

                resolved.relation_to_provisional_change
                == "supports"
            )


        elif observation.orientation not in [

            NEUTRAL,
            UNCLEAR

        ]:

            assert (

                resolved.relation_to_provisional_change
                == "contradicts"
            )


# %% ============================================================
# Explicit hysteretic updater
#
# Same as 04.
# ================================================================

def update_change_tracker(
    state_before: dict,
    observation: SemanticObservation,
    resolved: ResolvedInterpretation,
    evidence_id: str,
    step: int
) -> tuple[dict, dict]:

    state_after = deepcopy(
        state_before
    )


    domain = (
        state_after[
            STATE_KEY
        ]
    )


    tracker = (
        domain[
            "change_tracker"
        ]
    )


    previous_status = (
        tracker[
            "status"
        ]
    )


    old_score = (
        tracker[
            "score"
        ]
    )


    decayed_score = (

        old_score
        * SCORE_DECAY
    )


    positive_signal = 0.0

    recovery_signal = 0.0


    # ------------------------------------------------------------
    # Change-supporting evidence
    # ------------------------------------------------------------

    if (

        resolved.relation_to_established_state
        == "contradicts"

        and

        resolved.candidate_change_direction
        == "reduced_outdoor_engagement"

    ):

        positive_signal = (

            resolved.evidence_for_longitudinal_change
        )


        tracker[
            "qualifying_evidence_count"
        ] += 1


        if (

            tracker[
                "first_signal_step"
            ]
            is None

        ):

            tracker[
                "first_signal_step"
            ] = step


        tracker[
            "last_signal_step"
        ] = step


        tracker[
            "provenance_ids"
        ].append(
            evidence_id
        )


    # ------------------------------------------------------------
    # Baseline-supporting evidence
    # ------------------------------------------------------------

    if (

        resolved.relation_to_established_state
        == "supports"

    ):

        recovery_signal += (

            BASELINE_SUPPORT_STRENGTH

            *

            observation.semantic_strength

            *

            observation.confidence
        )


        tracker[
            "baseline_support_count"
        ] += 1


        domain[
            "established_state"
        ][
            "last_confirmed_step"
        ] = step


    # ------------------------------------------------------------
    # Contradiction of provisional state
    # ------------------------------------------------------------

    if (

        resolved.relation_to_provisional_change
        == "contradicts"

    ):

        recovery_signal += (

            PROVISIONAL_CONTRADICTION_BONUS

            *

            observation.semantic_strength

            *

            observation.confidence
        )


    # ------------------------------------------------------------
    # Score update
    # ------------------------------------------------------------

    new_score = (

        decayed_score

        + positive_signal

        - recovery_signal
    )


    new_score = max(

        0.0,

        min(
            MAX_CHANGE_SCORE,
            new_score
        )
    )


    tracker[
        "score"
    ] = new_score


    tracker[
        "peak_score"
    ] = max(

        tracker[
            "peak_score"
        ],

        new_score
    )


    change_evidence_strength = min(

        1.0,

        new_score
        / ACTIVATION_THRESHOLD
    )


    tracker[
        "change_evidence_strength"
    ] = (
        change_evidence_strength
    )


    activation_crossed = False

    deactivation_crossed = False

    hysteresis_hold = False


    # ------------------------------------------------------------
    # Previously active
    # ------------------------------------------------------------

    if previous_status == "active":

        if (

            new_score
            >= DEACTIVATION_THRESHOLD

        ):

            tracker[
                "status"
            ] = "active"


            if (

                new_score
                < ACTIVATION_THRESHOLD

            ):

                hysteresis_hold = True


        elif (

            new_score
            >= MONITORING_THRESHOLD

        ):

            tracker[
                "status"
            ] = "monitoring"


            deactivation_crossed = True


            tracker[
                "last_deactivation_step"
            ] = step


        else:

            tracker[
                "status"
            ] = "inactive"


            deactivation_crossed = True


            tracker[
                "last_deactivation_step"
            ] = step


    # ------------------------------------------------------------
    # Previously not active
    # ------------------------------------------------------------

    else:

        if (

            new_score
            >= ACTIVATION_THRESHOLD

            and

            tracker[
                "qualifying_evidence_count"
            ]
            >= MIN_EVIDENCE_FOR_ACTIVATION

        ):

            tracker[
                "status"
            ] = "active"


            activation_crossed = True


            if (

                tracker[
                    "activation_step"
                ]
                is None

            ):

                tracker[
                    "activation_step"
                ] = step


        elif (

            new_score
            >= MONITORING_THRESHOLD

        ):

            tracker[
                "status"
            ] = "monitoring"


        else:

            tracker[
                "status"
            ] = "inactive"


    # ------------------------------------------------------------
    # Provisional state
    # ------------------------------------------------------------

    if (

        tracker[
            "status"
        ]
        == "active"

    ):

        first_step = (

            tracker[
                "first_signal_step"
            ]
        )


        duration_steps = (

            step
            - first_step
            + 1

            if first_step is not None

            else 1
        )


        domain[
            "provisional_change"
        ] = {

            "active":
                True,

            "candidate":
                (
                    "Reduced interest in morning walks "
                    "and in going outside."
                ),

            "orientation":
                AWAY_FROM_OUTDOOR,

            "direction":
                "reduced_outdoor_engagement",

            "change_evidence_strength":
                change_evidence_strength,

            "duration_steps":
                duration_steps,

            "evidence_count":
                tracker[
                    "qualifying_evidence_count"
                ],

            "activation_step":
                tracker[
                    "activation_step"
                ],

            "provenance_ids":
                list(
                    tracker[
                        "provenance_ids"
                    ]
                )
        }


    else:

        domain[
            "provisional_change"
        ] = None


    transition = {

        "previous_status":
            previous_status,

        "new_status":
            tracker[
                "status"
            ],

        "old_score":
            old_score,

        "decayed_score":
            decayed_score,

        "positive_signal":
            positive_signal,

        "recovery_signal":
            recovery_signal,

        "new_score":
            new_score,

        "change_evidence_strength":
            change_evidence_strength,

        "activation_crossed":
            activation_crossed,

        "deactivation_crossed":
            deactivation_crossed,

        "hysteresis_hold":
            hysteresis_hold
    }


    return (
        state_after,
        transition
    )


# %% ============================================================
# Diagnostic alpha_t
# ================================================================

def compute_revision_openness(
    state_after: dict,
    resolved: ResolvedInterpretation
) -> float:

    domain = (
        state_after[
            STATE_KEY
        ]
    )


    established_stability = (

        domain[
            "established_state"
        ][
            "stability"
        ]
    )


    tracker = (

        domain[
            "change_tracker"
        ]
    )


    semantic_change_signal = (

        resolved.evidence_for_longitudinal_change
    )


    accumulated_signal = min(

        1.0,

        tracker[
            "score"
        ]
        / ACTIVATION_THRESHOLD
    )


    raw_openness = (

        0.40
        * semantic_change_signal

        +

        0.60
        * accumulated_signal
    )


    resistance = (

        0.65
        * established_stability
    )


    alpha_t = (

        raw_openness

        * (1.0 - resistance)
    )


    return max(

        0.0,

        min(
            1.0,
            alpha_t
        )
    )


# %% ============================================================
# JSONL utilities
# ================================================================

def append_jsonl(
    path: Path,
    record: dict
):

    with path.open(
        "a",
        encoding="utf-8"
    ) as f:

        f.write(

            json.dumps(
                record,
                ensure_ascii=False
            )

            + "\n"
        )


def load_jsonl(
    path: Path
) -> list[dict]:

    if not path.exists():

        return []


    records = []


    with path.open(
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            line = line.strip()


            if not line:

                continue


            records.append(
                json.loads(
                    line
                )
            )


    return records


# %% ============================================================
# Optional full reset
# ================================================================

if RESET_ALL:

    for path in RUN_DIR.glob(
        "run_*.jsonl"
    ):

        path.unlink()


    for path in [

        METRICS_CSV,
        SUMMARY_JSON

    ]:

        if path.exists():

            path.unlink()


    print(
        "\nAll previous replication logs cleared."
    )


# %% ============================================================
# Validate resumed records
# ================================================================

def validate_existing_records(
    records: list[dict]
):

    if not records:

        return


    first = records[0]


    if (

        first.get(
            "experiment_version"
        )
        != EXPERIMENT_VERSION

    ):

        raise RuntimeError(
            "Existing run log has a different experiment version. "
            "Set RESET_ALL=True before continuing."
        )


    if (

        first.get(
            "scenario_hash"
        )
        != SCENARIO_HASH

    ):

        raise RuntimeError(
            "Existing run log belongs to a different scenario. "
            "Set RESET_ALL=True before continuing."
        )


# %% ============================================================
# Run one replication
# ================================================================

def run_replication(
    run_id: int
) -> list[dict]:

    log_path = (

        RUN_DIR
        / f"run_{run_id:03d}.jsonl"
    )


    records = (
        load_jsonl(
            log_path
        )
    )


    validate_existing_records(
        records
    )


    if len(records) == 200:

        print(
            f"\nRun {run_id:03d}: already complete."
        )

        return records


    if len(records) > 200:

        raise RuntimeError(
            f"Run {run_id:03d} contains more than 200 records."
        )


    if records:

        state = deepcopy(

            records[-1][
                "state_after"
            ]
        )


        start_index = len(
            records
        )


        print(
            f"\nRun {run_id:03d}: "
            f"resuming from step {start_index + 1}."
        )


    else:

        state = deepcopy(
            INITIAL_STATE
        )


        start_index = 0


        print(
            f"\nRun {run_id:03d}: starting."
        )


    for index in range(
        start_index,
        200
    ):

        event = (
            SCENARIO[
                index
            ]
        )


        step = event[
            "step"
        ]


        phase = event[
            "phase"
        ]


        utterance = event[
            "utterance"
        ]


        expected_orientation = (
            event[
                "expected_orientation"
            ]
        )


        evidence_id = (

            f"run_{run_id:03d}"
            f"_step_{step:03d}"
        )


        state_before = deepcopy(
            state
        )


        # --------------------------------------------------------
        # 1. LLM semantic extraction
        # --------------------------------------------------------

        observation, usage = (
            parse_semantic_observation(
                utterance
            )
        )


        # --------------------------------------------------------
        # 2. Deterministic state-relative interpretation
        # --------------------------------------------------------

        resolved = (
            resolve_interpretation(

                state_before,

                observation
            )
        )


        assert_interface_invariants(

            state_before,

            observation,

            resolved
        )


        semantic_correct = (

            observation.orientation
            == expected_orientation
        )


        # --------------------------------------------------------
        # 3. Explicit state transition
        # --------------------------------------------------------

        state_after, transition = (
            update_change_tracker(

                state_before,

                observation,

                resolved,

                evidence_id,

                step
            )
        )


        # --------------------------------------------------------
        # 4. Diagnostic alpha
        # --------------------------------------------------------

        alpha_t = (
            compute_revision_openness(

                state_after,

                resolved
            )
        )


        state = state_after


        tracker = (

            state_after[
                STATE_KEY
            ][
                "change_tracker"
            ]
        )


        provisional = (

            state_after[
                STATE_KEY
            ][
                "provisional_change"
            ]
        )


        # --------------------------------------------------------
        # 5. Full trace
        # --------------------------------------------------------

        record = {

            "experiment_version":
                EXPERIMENT_VERSION,

            "scenario_hash":
                SCENARIO_HASH,

            "run_id":
                run_id,

            "step":
                step,

            "phase":
                phase,

            "utterance":
                utterance,

            "expected_orientation":
                expected_orientation,

            "semantic_observation":
                observation.model_dump(),

            "semantic_correct":
                semantic_correct,

            "resolved_interpretation":
                resolved.model_dump(),

            "state_before":
                state_before,

            "transition":
                transition,

            "alpha_t":
                alpha_t,

            "state_after":
                state_after,

            "summary": {

                "change_score":
                    tracker[
                        "score"
                    ],

                "change_evidence_strength":
                    tracker[
                        "change_evidence_strength"
                    ],

                "tracker_status":
                    tracker[
                        "status"
                    ],

                "provisional_active":
                    (
                        provisional
                        is not None
                    ),

                "hysteresis_hold":
                    transition[
                        "hysteresis_hold"
                    ]
            },

            "usage":
                usage
        }


        records.append(
            record
        )


        append_jsonl(
            log_path,
            record
        )


        # --------------------------------------------------------
        # Console trace
        # --------------------------------------------------------

        important_transition = (

            transition[
                "activation_crossed"
            ]

            or

            transition[
                "deactivation_crossed"
            ]
        )


        if (

            step % PRINT_EVERY == 0

            or

            important_transition

            or

            not semantic_correct

        ):

            print(
                f"  run {run_id:03d} "
                f"step {step:03d}/200 "
                f"[{phase}]"
            )


            print(
                "      semantic :",
                observation.orientation
            )


            print(
                "      correct  :",
                semantic_correct
            )


            print(
                "      score    :",
                round(
                    tracker[
                        "score"
                    ],
                    3
                )
            )


            print(
                "      status   :",
                tracker[
                    "status"
                ]
            )


            if transition[
                "activation_crossed"
            ]:

                print(
                    "      *** ACTIVATED ***"
                )


            if transition[
                "deactivation_crossed"
            ]:

                print(
                    "      *** DEACTIVATED ***"
                )


    assert len(records) == 200


    print(
        f"Run {run_id:03d}: complete."
    )


    return records


# %% ============================================================
# Run all replications
# ================================================================

all_runs = []


for run_id in range(
    N_RUNS
):

    run_records = (
        run_replication(
            run_id
        )
    )


    all_runs.append(
        run_records
    )


# %% ============================================================
# Phase definitions
# ================================================================

PHASE_ORDER = [

    "stable_baseline",

    "transient_perturbation",

    "restabilization",

    "genuine_change",

    "mixed_evidence",

    "recovery"
]


PHASE_RANGES = {

    "stable_baseline":
        (1, 60),

    "transient_perturbation":
        (61, 65),

    "restabilization":
        (66, 100),

    "genuine_change":
        (101, 145),

    "mixed_evidence":
        (146, 160),

    "recovery":
        (161, 200)
}


# %% ============================================================
# Per-run scalar metrics
# ================================================================

def compute_run_metrics(
    records: list[dict]
) -> dict:

    semantic_accuracy = np.mean(

        [
            record[
                "semantic_correct"
            ]
            for record in records
        ]
    )


    semantic_error_count = sum(

        not record[
            "semantic_correct"
        ]

        for record in records
    )


    pre_change = [

        record

        for record in records

        if record[
            "step"
        ] <= 100
    ]


    false_activation = any(

        record[
            "summary"
        ][
            "provisional_active"
        ]

        for record in pre_change
    )


    transient_records = [

        record

        for record in records

        if record[
            "phase"
        ] == "transient_perturbation"
    ]


    transient_peak_score = max(

        record[
            "summary"
        ][
            "change_score"
        ]

        for record in transient_records
    )


    genuine_change_records = [

        record

        for record in records

        if record[
            "phase"
        ] == "genuine_change"
    ]


    activation_candidates = [

        record[
            "step"
        ]

        for record in genuine_change_records

        if record[
            "transition"
        ][
            "activation_crossed"
        ]
    ]


    if activation_candidates:

        first_activation_step = (
            activation_candidates[0]
        )


        activation_latency = (

            first_activation_step
            - 101
            + 1
        )


    else:

        first_activation_step = np.nan

        activation_latency = np.nan


    recovery_records = [

        record

        for record in records

        if record[
            "phase"
        ] == "recovery"
    ]


    active_on_first_recovery = (

        recovery_records[0][
            "summary"
        ][
            "provisional_active"
        ]
    )


    deactivation_candidates = [

        record[
            "step"
        ]

        for record in recovery_records

        if record[
            "transition"
        ][
            "deactivation_crossed"
        ]
    ]


    if deactivation_candidates:

        first_deactivation_step = (
            deactivation_candidates[0]
        )


        recovery_latency = (

            first_deactivation_step
            - 161
            + 1
        )


    else:

        first_deactivation_step = np.nan

        recovery_latency = np.nan


    mixed_records = [

        record

        for record in records

        if record[
            "phase"
        ] == "mixed_evidence"
    ]


    mixed_status_switches = 0


    for previous, current in zip(

        mixed_records[:-1],

        mixed_records[1:]

    ):

        if (

            previous[
                "summary"
            ][
                "tracker_status"
            ]

            !=

            current[
                "summary"
            ][
                "tracker_status"
            ]

        ):

            mixed_status_switches += 1


    end_active = (

        records[-1][
            "summary"
        ][
            "provisional_active"
        ]
    )


    peak_score = max(

        record[
            "summary"
        ][
            "change_score"
        ]

        for record in records
    )


    total_tokens = sum(

        record[
            "usage"
        ].get(
            "total_tokens",
            0
        )

        for record in records
    )


    return {

        "run_id":
            records[0][
                "run_id"
            ],

        "semantic_accuracy":
            semantic_accuracy,

        "semantic_error_count":
            semantic_error_count,

        "false_activation":
            false_activation,

        "transient_peak_score":
            transient_peak_score,

        "first_activation_step":
            first_activation_step,

        "activation_latency":
            activation_latency,

        "active_on_first_recovery":
            active_on_first_recovery,

        "first_deactivation_step":
            first_deactivation_step,

        "recovery_latency":
            recovery_latency,

        "mixed_status_switches":
            mixed_status_switches,

        "end_active":
            end_active,

        "peak_score":
            peak_score,

        "total_tokens":
            total_tokens
    }


# %% ============================================================
# Collect scalar metrics
# ================================================================

metrics_rows = [

    compute_run_metrics(
        records
    )

    for records in all_runs
]


metrics_df = pd.DataFrame(
    metrics_rows
)


metrics_df.to_csv(
    METRICS_CSV,
    index=False
)


# %% ============================================================
# Trajectory matrices
#
# shape:
#
#     N_RUNS × 200 timesteps
# ================================================================

def trajectory_matrix(
    key: str
) -> np.ndarray:

    matrix = []


    for records in all_runs:

        if key == "change_score":

            values = [

                record[
                    "summary"
                ][
                    "change_score"
                ]

                for record in records
            ]


        elif key == "alpha_t":

            values = [

                record[
                    "alpha_t"
                ]

                for record in records
            ]


        elif key == "provisional_active":

            values = [

                1.0

                if record[
                    "summary"
                ][
                    "provisional_active"
                ]

                else 0.0

                for record in records
            ]


        elif key == "semantic_error":

            values = [

                0.0

                if record[
                    "semantic_correct"
                ]

                else 1.0

                for record in records
            ]


        else:

            raise ValueError(
                key
            )


        matrix.append(
            values
        )


    return np.asarray(
        matrix,
        dtype=float
    )


change_matrix = (
    trajectory_matrix(
        "change_score"
    )
)


alpha_matrix = (
    trajectory_matrix(
        "alpha_t"
    )
)


active_matrix = (
    trajectory_matrix(
        "provisional_active"
    )
)


semantic_error_matrix = (
    trajectory_matrix(
        "semantic_error"
    )
)


# %% ============================================================
# Bootstrap median 95% CI
#
# This is a confidence interval for the median trajectory across
# runs, NOT a 95% prediction interval for individual runs.
# ================================================================

def bootstrap_median_ci(
    matrix: np.ndarray,
    n_bootstrap: int,
    seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    rng = np.random.default_rng(
        seed
    )


    n_runs = matrix.shape[0]

    n_steps = matrix.shape[1]


    observed_median = np.median(
        matrix,
        axis=0
    )


    bootstrap_medians = np.empty(
        (
            n_bootstrap,
            n_steps
        ),
        dtype=float
    )


    for b in range(
        n_bootstrap
    ):

        indices = rng.integers(

            low=0,

            high=n_runs,

            size=n_runs
        )


        sampled = matrix[
            indices,
            :
        ]


        bootstrap_medians[
            b,
            :
        ] = np.median(

            sampled,

            axis=0
        )


    lower = np.quantile(

        bootstrap_medians,

        0.025,

        axis=0
    )


    upper = np.quantile(

        bootstrap_medians,

        0.975,

        axis=0
    )


    return (
        observed_median,
        lower,
        upper
    )


change_median, change_low, change_high = (
    bootstrap_median_ci(

        change_matrix,

        BOOTSTRAP_SAMPLES,

        BOOTSTRAP_SEED
    )
)


alpha_median, alpha_low, alpha_high = (
    bootstrap_median_ci(

        alpha_matrix,

        BOOTSTRAP_SAMPLES,

        BOOTSTRAP_SEED + 1
    )
)


# %% ============================================================
# Wilson interval for fraction active
# ================================================================

def wilson_interval(
    successes: np.ndarray,
    n: int,
    z: float = 1.96
) -> tuple[np.ndarray, np.ndarray]:

    p = (

        successes
        / n
    )


    denominator = (

        1.0
        + z**2 / n
    )


    center = (

        p
        + z**2 / (2.0 * n)

    ) / denominator


    margin = (

        z

        * np.sqrt(

            (
                p * (1.0 - p)
                + z**2 / (4.0 * n)
            )

            / n
        )

    ) / denominator


    lower = np.maximum(
        0.0,
        center - margin
    )


    upper = np.minimum(
        1.0,
        center + margin
    )


    return (
        lower,
        upper
    )


active_count = np.sum(
    active_matrix,
    axis=0
)


active_fraction = (

    active_count
    / N_RUNS
)


active_low, active_high = (
    wilson_interval(

        active_count,

        N_RUNS
    )
)


# %% ============================================================
# Semantic error rate by timestep
# ================================================================

semantic_error_rate = np.mean(

    semantic_error_matrix,

    axis=0
)


# %% ============================================================
# Phase boundaries
# ================================================================

PHASE_BOUNDARIES = [

    60.5,
    65.5,
    100.5,
    145.5,
    160.5
]


def add_phase_boundaries():

    for boundary in PHASE_BOUNDARIES:

        plt.axvline(

            boundary,

            linestyle="--",

            alpha=0.30
        )


steps = np.arange(
    1,
    201
)


# %% ============================================================
# Plot 1
# Median change score + bootstrap 95% CI
# ================================================================

plt.figure(
    figsize=(14, 6)
)


plt.plot(

    steps,

    change_median,

    label="Median"
)


plt.fill_between(

    steps,

    change_low,

    change_high,

    alpha=0.20,

    label="Bootstrap 95% CI"
)


plt.axhline(

    ACTIVATION_THRESHOLD,

    linestyle="--",

    label="Activation threshold"
)


plt.axhline(

    DEACTIVATION_THRESHOLD,

    linestyle=":",

    label="Deactivation threshold"
)


add_phase_boundaries()


plt.xlabel(
    "Synthetic day"
)


plt.ylabel(
    "Accumulated change score"
)


plt.title(
    "Replication: slow-state trajectory"
)


plt.legend()


plt.tight_layout()


plt.savefig(

    OUTPUT_DIR
    / "replication_change_score.png",

    dpi=200
)


plt.show()


# %% ============================================================
# Plot 2
# Median alpha + bootstrap 95% CI
# ================================================================

plt.figure(
    figsize=(14, 6)
)


plt.plot(

    steps,

    alpha_median,

    label="Median"
)


plt.fill_between(

    steps,

    alpha_low,

    alpha_high,

    alpha=0.20,

    label="Bootstrap 95% CI"
)


add_phase_boundaries()


plt.xlabel(
    "Synthetic day"
)


plt.ylabel(
    "Revision openness alpha_t"
)


plt.title(
    "Replication: revision openness"
)


plt.legend()


plt.tight_layout()


plt.savefig(

    OUTPUT_DIR
    / "replication_alpha.png",

    dpi=200
)


plt.show()


# %% ============================================================
# Plot 3
# Fraction of runs with provisional state active
# ================================================================

plt.figure(
    figsize=(14, 5)
)


plt.plot(

    steps,

    active_fraction,

    label="Fraction active"
)


plt.fill_between(

    steps,

    active_low,

    active_high,

    alpha=0.20,

    label="Wilson 95% interval"
)


add_phase_boundaries()


plt.xlabel(
    "Synthetic day"
)


plt.ylabel(
    "Fraction of runs active"
)


plt.title(
    "Replication: provisional-state activation"
)


plt.ylim(
    -0.05,
    1.05
)


plt.legend()


plt.tight_layout()


plt.savefig(

    OUTPUT_DIR
    / "replication_provisional_active.png",

    dpi=200
)


plt.show()


# %% ============================================================
# Plot 4
# Semantic error rate by timestep
#
# Particularly useful for finding unstable individual utterances.
# ================================================================

plt.figure(
    figsize=(14, 5)
)


plt.plot(

    steps,

    semantic_error_rate
)


add_phase_boundaries()


plt.xlabel(
    "Synthetic day"
)


plt.ylabel(
    "Semantic error rate across runs"
)


plt.title(
    "Replication: semantic parser error rate by timestep"
)


plt.ylim(
    -0.02,
    1.02
)


plt.tight_layout()


plt.savefig(

    OUTPUT_DIR
    / "semantic_error_rate_by_timestep.png",

    dpi=200
)


plt.show()


# %% ============================================================
# Scalar summary helpers
# ================================================================

def numeric_summary(
    series: pd.Series
) -> dict:

    values = (

        series
        .dropna()
        .astype(float)
        .to_numpy()
    )


    if len(values) == 0:

        return {

            "n":
                0,

            "median":
                None,

            "q025":
                None,

            "q975":
                None
        }


    return {

        "n":
            int(
                len(values)
            ),

        "median":
            float(
                np.median(
                    values
                )
            ),

        "q025":
            float(
                np.quantile(
                    values,
                    0.025
                )
            ),

        "q975":
            float(
                np.quantile(
                    values,
                    0.975
                )
            )
    }


# %% ============================================================
# Replication-level summary statistics
# ================================================================

false_activation_rate = float(

    metrics_df[
        "false_activation"
    ].mean()
)


detection_success_rate = float(

    metrics_df[
        "activation_latency"
    ].notna().mean()
)


recovery_success_rate = float(

    metrics_df[
        "recovery_latency"
    ].notna().mean()
)


final_inactive_rate = float(

    (
        ~metrics_df[
            "end_active"
        ]
    ).mean()
)


first_recovery_residue_rate = float(

    metrics_df[
        "active_on_first_recovery"
    ].mean()
)


semantic_accuracy_median = float(

    metrics_df[
        "semantic_accuracy"
    ].median()
)


semantic_accuracy_min = float(

    metrics_df[
        "semantic_accuracy"
    ].min()
)


summary = {

    "experiment_version":
        EXPERIMENT_VERSION,

    "model":
        MODEL,

    "n_runs":
        N_RUNS,

    "timesteps_per_run":
        200,

    "scenario_seed":
        SCENARIO_SEED,

    "scenario_hash":
        SCENARIO_HASH,

    "semantic_accuracy": {

        "median":
            semantic_accuracy_median,

        "minimum":
            semantic_accuracy_min
    },

    "false_activation_rate":
        false_activation_rate,

    "detection_success_rate":
        detection_success_rate,

    "recovery_success_rate":
        recovery_success_rate,

    "first_recovery_residue_rate":
        first_recovery_residue_rate,

    "final_inactive_rate":
        final_inactive_rate,

    "activation_latency":
        numeric_summary(
            metrics_df[
                "activation_latency"
            ]
        ),

    "recovery_latency":
        numeric_summary(
            metrics_df[
                "recovery_latency"
            ]
        ),

    "transient_peak_score":
        numeric_summary(
            metrics_df[
                "transient_peak_score"
            ]
        ),

    "mixed_status_switches":
        numeric_summary(
            metrics_df[
                "mixed_status_switches"
            ]
        ),

    "semantic_error_count":
        numeric_summary(
            metrics_df[
                "semantic_error_count"
            ]
        ),

    "tokens_per_run":
        numeric_summary(
            metrics_df[
                "total_tokens"
            ]
        ),

    "total_tokens_all_runs":
        int(
            metrics_df[
                "total_tokens"
            ].sum()
        )
}


with SUMMARY_JSON.open(
    "w",
    encoding="utf-8"
) as f:

    json.dump(

        summary,

        f,

        indent=2,

        ensure_ascii=False
    )


# %% ============================================================
# Most unstable semantic timesteps
# ================================================================

unstable_steps = []


for step_index in range(
    200
):

    rate = (
        semantic_error_rate[
            step_index
        ]
    )


    if rate > 0:

        event = (
            SCENARIO[
                step_index
            ]
        )


        unstable_steps.append(

            {

                "step":
                    step_index + 1,

                "phase":
                    event[
                        "phase"
                    ],

                "utterance":
                    event[
                        "utterance"
                    ],

                "expected_orientation":
                    event[
                        "expected_orientation"
                    ],

                "error_rate":
                    float(
                        rate
                    )
            }
        )


unstable_steps = sorted(

    unstable_steps,

    key=lambda item:
        item[
            "error_rate"
        ],

    reverse=True
)


# %% ============================================================
# Console report
# ================================================================

print("\n")
print("=" * 80)

print(
    "REPLICATION SUMMARY"
)

print("=" * 80)


print(
    "\nRuns:",
    N_RUNS
)


print(
    "Timesteps/run:",
    200
)


print(
    "\nSemantic accuracy median:",
    round(
        semantic_accuracy_median,
        4
    )
)


print(
    "Semantic accuracy minimum:",
    round(
        semantic_accuracy_min,
        4
    )
)


print(
    "\nFalse activation rate:",
    round(
        false_activation_rate,
        4
    )
)


print(
    "Change detection success rate:",
    round(
        detection_success_rate,
        4
    )
)


print(
    "Recovery success rate:",
    round(
        recovery_success_rate,
        4
    )
)


print(
    "Residue on first recovery observation:",
    round(
        first_recovery_residue_rate,
        4
    )
)


print(
    "Final inactive rate:",
    round(
        final_inactive_rate,
        4
    )
)


print(
    "\nActivation latency:"
)


print(
    summary[
        "activation_latency"
    ]
)


print(
    "\nRecovery latency:"
)


print(
    summary[
        "recovery_latency"
    ]
)


print(
    "\nMixed-phase status switches:"
)


print(
    summary[
        "mixed_status_switches"
    ]
)


print(
    "\nSemantic errors/run:"
)


print(
    summary[
        "semantic_error_count"
    ]
)


print(
    "\nTotal token usage across all runs:",
    summary[
        "total_tokens_all_runs"
    ]
)


# %% ============================================================
# Print semantic instability hotspots
# ================================================================

print("\n")
print("=" * 80)

print(
    "SEMANTIC INSTABILITY HOTSPOTS"
)

print("=" * 80)


if len(
    unstable_steps
) == 0:

    print(
        "\nNo semantic classification errors across any replication."
    )


else:

    for item in unstable_steps:

        print(
            f"\nStep {item['step']}"
        )


        print(
            "phase      :",
            item[
                "phase"
            ]
        )


        print(
            "error rate :",
            round(
                item[
                    "error_rate"
                ],
                3
            )
        )


        print(
            "expected   :",
            item[
                "expected_orientation"
            ]
        )


        print(
            "utterance  :",
            item[
                "utterance"
            ]
        )


# %% ============================================================
# Preliminary acceptance criteria
#
# These are engineering criteria for this toy experiment.
# They are NOT scientific effect-size criteria.
# ================================================================

criterion_semantic_median = (

    semantic_accuracy_median
    >= 0.95
)


criterion_semantic_min = (

    semantic_accuracy_min
    >= 0.90
)


criterion_false_activation = (

    false_activation_rate
    <= 0.10
)


criterion_detection = (

    detection_success_rate
    >= 0.90
)


criterion_recovery = (

    recovery_success_rate
    >= 0.90
)


criterion_residue = (

    first_recovery_residue_rate
    >= 0.90
)


criterion_final_inactive = (

    final_inactive_rate
    >= 0.90
)


criterion_mixed_stability = (

    metrics_df[
        "mixed_status_switches"
    ].median()
    <= 1
)


all_pass = all(

    [

        criterion_semantic_median,

        criterion_semantic_min,

        criterion_false_activation,

        criterion_detection,

        criterion_recovery,

        criterion_residue,

        criterion_final_inactive,

        criterion_mixed_stability
    ]
)


print("\n")
print("=" * 80)

print(
    "REPLICATION ACCEPTANCE CHECK"
)

print("=" * 80)


print(
    "\nMedian semantic accuracy >= .95:",
    criterion_semantic_median
)


print(
    "Worst-run semantic accuracy >= .90:",
    criterion_semantic_min
)


print(
    "False activation rate <= .10:",
    criterion_false_activation
)


print(
    "Change detection success >= .90:",
    criterion_detection
)


print(
    "Recovery success >= .90:",
    criterion_recovery
)


print(
    "History residue on recovery >= .90:",
    criterion_residue
)


print(
    "Final inactive rate >= .90:",
    criterion_final_inactive
)


print(
    "Median mixed-phase switches <= 1:",
    criterion_mixed_stability
)


if all_pass:

    print(
        "\nPASS:"
        "\nThe explicit slow-state architecture reproduces its "
        "qualitative longitudinal dynamics across stochastic "
        "LLM replications of the same 200-step environment."
    )


else:

    print(
        "\nNOT CLEAN:"
        "\nDo not tune state thresholds yet. "
        "Inspect run-level JSONL traces and determine whether "
        "failures originate in semantic extraction or state dynamics."
    )


print(
    "\nMetrics CSV:"
)

print(
    METRICS_CSV
)


print(
    "\nSummary JSON:"
)

print(
    SUMMARY_JSON
)


print(
    "\nOutputs:"
)

print(
    OUTPUT_DIR
)