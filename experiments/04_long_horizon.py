# -*- coding: utf-8 -*-
"""
continuoue time recovery eval. 장기동역학 테스트, 중요한 것은 failure mode. 
"""

# %% ============================================================
# 04_long_horizon.py
#
# Long-horizon stress test for the explicit slow-state architecture.
#
# 1 timestep = 1 synthetic day
#
# Timeline:
#
#   1–60      stable baseline
#   61–65     transient perturbation
#   66–100    restabilization
#   101–145   genuine sustained change
#   146–160   mixed/conflicting evidence
#   161–200   sustained recovery
#
#
# Architecture:
#
#   utterance x_t
#       ↓
#   LLM semantic parser
#       ↓
#   semantic observation s_t
#       ↓
#   deterministic relation resolver + g_(t-1)
#       ↓
#   resolved interpretation z_t
#       ↓
#   explicit hysteretic updater
#       ↓
#   persistent state g_t
#
#
# Goals:
#
#   - no false activation during long stable history
#   - transient perturbation does not overwrite baseline
#   - genuine sustained change activates provisional state
#   - mixed evidence does not cause pathological oscillation
#   - recovery leaves residue but eventually resolves
#   - semantic errors can be localized independently
#
#
# IMPORTANT:
#
# This remains a proof-of-mechanism / engineering stress test.
# Update-law numbers are not scientifically calibrated.
# ================================================================


# %% Imports

from pathlib import Path
from copy import deepcopy

import os
import json
import random

from typing import Literal

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

import matplotlib.pyplot as plt


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
# Output configuration
# ================================================================

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "04_long_horizon"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


LOG_PATH = (
    OUTPUT_DIR
    / "long_horizon.jsonl"
)


METRICS_PATH = (
    OUTPUT_DIR
    / "long_horizon_metrics.json"
)


PROBE_PATH = (
    OUTPUT_DIR
    / "checkpoint_probe_comparison.json"
)


# ------------------------------------------------------------
# If False:
#     resume from existing JSONL if present.
#
# If True:
#     delete previous log and start from timestep 1.
#
# IMPORTANT:
# Set RESET_RUN=True if you change:
#   - scenarios
#   - thresholds
#   - update law
#   - semantic schema
# ------------------------------------------------------------

RESET_RUN = False


# ------------------------------------------------------------
# Avoid flooding Spyder console.
# Full information remains in JSONL.
# ------------------------------------------------------------

PRINT_EVERY = 10


# %% ============================================================
# Reproducibility
#
# Controls synthetic utterance selection.
#
# It does NOT control stochasticity inside the API model.
# ================================================================

SCENARIO_SEED = 42

rng = random.Random(
    SCENARIO_SEED
)


# %% ============================================================
# State-dynamics parameters
#
# Kept aligned with the refactored 03 implementation.
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
# LLM semantic observation
# ================================================================

class SemanticObservation(BaseModel):

    interpretation: str = Field(
        description=(
            "Concise description of what the utterance itself "
            "expresses, without comparing it to persistent state."
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
# State-relative interpretation
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

            "value": (
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
# All utterances are deliberately unambiguous enough that
# synthetic semantic ground truth can be assigned.
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

    "The weather is awful today, so I'm staying inside this morning."
]


CHANGE_BANK = [

    "I've been skipping my morning walk a lot lately.",

    "I haven't really wanted to go outside lately.",

    "I'd rather stay inside in the mornings now.",

    "The morning walk doesn't appeal to me much anymore.",

    "I've continued skipping the morning walk.",

    "Staying indoors has felt preferable lately.",

    "I don't really enjoy going outside in the morning anymore.",

    "I've been avoiding my usual morning walk recently.",

    "I keep choosing to stay inside instead of taking the walk.",

    "I don't think I want to keep doing the morning walk right now.",

    "My interest in going outside in the morning has really dropped.",

    "Lately I've preferred staying home instead of taking my walk."
]


RECOVERY_BANK = [

    "I've started enjoying my morning walks again.",

    "I took my morning walk today and it felt good.",

    "I've been wanting to get outside again lately.",

    "Going outside in the morning feels normal again.",

    "I want to keep doing my morning walks.",

    "I've returned to my usual morning walk routine.",

    "I've been enjoying getting outside again.",

    "My morning walks have been feeling good again lately.",

    "I've gone back to walking outside most mornings.",

    "I feel like my usual outdoor routine is returning."
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
# Build exactly 200 timesteps
# ================================================================

def build_long_horizon_scenario() -> list[dict]:

    events = []

    step = 1


    # ------------------------------------------------------------
    # Phase 1: long stable baseline
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
    # Phase 2: transient perturbation
    # t = 61–65
    #
    # Alternating brief anomalies and baseline-consistent evidence.
    #
    # Designed to test:
    #
    #   "one bad patch should not rewrite the person"
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
    # Phase 3: restabilization
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
    # Phase 4: genuine sustained change
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
    # Phase 5: mixed / conflicting evidence
    # t = 146–160
    #
    # Alternating evidence tests whether the state oscillates
    # pathologically or retains history-dependent inertia.
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
    # Phase 6: sustained recovery
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
    build_long_horizon_scenario()
)


# %% ============================================================
# Semantic parser
#
# Critically:
# receives ONLY current utterance.
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
# Deterministic relation resolver
#
# g_(t-1) enters here.
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
# Explicit state updater
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


    # ------------------------------------------------------------
    # Passive decay
    # ------------------------------------------------------------

    decayed_score = (

        old_score
        * SCORE_DECAY
    )


    positive_signal = 0.0

    recovery_signal = 0.0


    # ------------------------------------------------------------
    # Candidate-change support
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
    # Baseline-supporting recovery
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
    # Active provisional contradiction bonus
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
    # Update score
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
    ] = change_evidence_strength


    # ------------------------------------------------------------
    # Hysteretic state machine
    # ------------------------------------------------------------

    activation_crossed = False

    deactivation_crossed = False

    hysteresis_hold = False


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
# Reset / resume
# ================================================================

if (

    RESET_RUN

    and

    LOG_PATH.exists()

):

    LOG_PATH.unlink()


records = load_jsonl(
    LOG_PATH
)


if len(records) == 0:

    state = deepcopy(
        INITIAL_STATE
    )

    start_index = 0


    print(
        "\nStarting new 200-step run."
    )


else:

    state = deepcopy(

        records[-1][
            "state_after"
        ]
    )


    start_index = len(
        records
    )


    print(
        f"\nResuming from timestep {start_index + 1}."
    )


# %% ============================================================
# Long-horizon run
# ================================================================

for index in range(
    start_index,
    len(SCENARIO)
):

    event = SCENARIO[
        index
    ]


    step = event[
        "step"
    ]


    phase = event[
        "phase"
    ]


    utterance = event[
        "utterance"
    ]


    expected_orientation = event[
        "expected_orientation"
    ]


    evidence_id = (
        f"long_horizon_step_{step:03d}"
    )


    state_before = deepcopy(
        state
    )


    # ------------------------------------------------------------
    # 1. Semantic extraction
    # ------------------------------------------------------------

    observation, usage = (
        parse_semantic_observation(
            utterance
        )
    )


    # ------------------------------------------------------------
    # 2. State-relative interpretation
    # ------------------------------------------------------------

    resolved = (
        resolve_interpretation(

            state_before,

            observation
        )
    )


    # ------------------------------------------------------------
    # 3. Interface invariants
    # ------------------------------------------------------------

    assert_interface_invariants(

        state_before,

        observation,

        resolved
    )


    # ------------------------------------------------------------
    # 4. Semantic synthetic-ground-truth check
    # ------------------------------------------------------------

    semantic_correct = (

        observation.orientation
        == expected_orientation
    )


    # ------------------------------------------------------------
    # 5. State transition
    # ------------------------------------------------------------

    state_after, transition = (
        update_change_tracker(

            state_before,

            observation,

            resolved,

            evidence_id,

            step
        )
    )


    # ------------------------------------------------------------
    # 6. alpha_t
    # ------------------------------------------------------------

    alpha_t = (
        compute_revision_openness(

            state_after,

            resolved
        )
    )


    state = state_after


    tracker = (

        state[
            STATE_KEY
        ][
            "change_tracker"
        ]
    )


    provisional = (

        state[
            STATE_KEY
        ][
            "provisional_change"
        ]
    )


    # ------------------------------------------------------------
    # 7. Log
    # ------------------------------------------------------------

    record = {

        "step":
            step,

        "synthetic_day":
            event[
                "synthetic_day"
            ],

        "phase":
            phase,

        "evidence_id":
            evidence_id,

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
        LOG_PATH,
        record
    )


    # ------------------------------------------------------------
    # Console reporting
    # ------------------------------------------------------------

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
            f"\n[{step:03d}/200] "
            f"[{phase}]"
        )


        print(
            "utterance        :",
            utterance
        )


        print(
            "semantic         :",
            observation.orientation
        )


        print(
            "semantic correct :",
            semantic_correct
        )


        print(
            "temporal scope   :",
            observation.temporal_scope
        )


        print(
            "score            :",
            round(
                tracker[
                    "score"
                ],
                3
            )
        )


        print(
            "status           :",
            tracker[
                "status"
            ]
        )


        print(
            "alpha_t          :",
            round(
                alpha_t,
                3
            )
        )


        if transition[
            "activation_crossed"
        ]:

            print(
                "*** PROVISIONAL STATE ACTIVATED ***"
            )


        if transition[
            "deactivation_crossed"
        ]:

            print(
                "*** PROVISIONAL STATE DEACTIVATED ***"
            )


# %% ============================================================
# Ensure full trajectory exists
# ================================================================

assert len(records) == 200


# %% ============================================================
# Phase helpers
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


def records_for_phase(
    phase: str
) -> list[dict]:

    return [

        record

        for record in records

        if record[
            "phase"
        ] == phase
    ]


# %% ============================================================
# Semantic accuracy
# ================================================================

semantic_correct_total = sum(

    int(
        record[
            "semantic_correct"
        ]
    )

    for record in records
)


semantic_accuracy_total = (

    semantic_correct_total
    / len(records)
)


semantic_accuracy_by_phase = {}


for phase in PHASE_ORDER:

    phase_records = (
        records_for_phase(
            phase
        )
    )


    phase_correct = sum(

        int(
            record[
                "semantic_correct"
            ]
        )

        for record in phase_records
    )


    semantic_accuracy_by_phase[
        phase
    ] = (

        phase_correct
        / len(phase_records)
    )


# %% ============================================================
# State-transition metrics
# ================================================================

pre_change_records = [

    record

    for record in records

    if record[
        "step"
    ] <= 100
]


false_activation_steps = [

    record[
        "step"
    ]

    for record in pre_change_records

    if record[
        "summary"
    ][
        "provisional_active"
    ]
]


genuine_change_records = (
    records_for_phase(
        "genuine_change"
    )
)


activation_steps = [

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


if activation_steps:

    first_activation_step = (
        activation_steps[0]
    )


    activation_latency = (

        first_activation_step
        - PHASE_RANGES[
            "genuine_change"
        ][0]
        + 1
    )


else:

    first_activation_step = None

    activation_latency = None


# ------------------------------------------------------------
# Recovery
# ------------------------------------------------------------

recovery_records = (
    records_for_phase(
        "recovery"
    )
)


active_on_first_recovery_step = (

    recovery_records[0][
        "summary"
    ][
        "provisional_active"
    ]
)


deactivation_steps = [

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


if deactivation_steps:

    first_deactivation_step = (
        deactivation_steps[0]
    )


    recovery_latency = (

        first_deactivation_step
        - PHASE_RANGES[
            "recovery"
        ][0]
        + 1
    )


else:

    first_deactivation_step = None

    recovery_latency = None


end_active = (

    records[-1][
        "summary"
    ][
        "provisional_active"
    ]
)


# ------------------------------------------------------------
# Mixed-phase stability
# ------------------------------------------------------------

mixed_records = (
    records_for_phase(
        "mixed_evidence"
    )
)


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


# ------------------------------------------------------------
# Peak
# ------------------------------------------------------------

peak_score = max(

    record[
        "summary"
    ][
        "change_score"
    ]

    for record in records
)


# %% ============================================================
# Token usage
# ================================================================

total_input_tokens = sum(

    record[
        "usage"
    ].get(
        "input_tokens",
        0
    )

    for record in records
)


total_output_tokens = sum(

    record[
        "usage"
    ].get(
        "output_tokens",
        0
    )

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


# %% ============================================================
# Save metrics
# ================================================================

metrics = {

    "n_timesteps":
        len(records),

    "semantic_accuracy_total":
        semantic_accuracy_total,

    "semantic_accuracy_by_phase":
        semantic_accuracy_by_phase,

    "false_activation_steps_before_genuine_change":
        false_activation_steps,

    "first_activation_step":
        first_activation_step,

    "activation_latency":
        activation_latency,

    "active_on_first_recovery_step":
        active_on_first_recovery_step,

    "first_deactivation_step":
        first_deactivation_step,

    "recovery_latency":
        recovery_latency,

    "end_active":
        end_active,

    "mixed_phase_status_switches":
        mixed_status_switches,

    "peak_change_score":
        peak_score,

    "token_usage": {

        "input_tokens":
            total_input_tokens,

        "output_tokens":
            total_output_tokens,

        "total_tokens":
            total_tokens
    }
}


with METRICS_PATH.open(
    "w",
    encoding="utf-8"
) as f:

    json.dump(

        metrics,

        f,

        indent=2,

        ensure_ascii=False
    )


# %% ============================================================
# Console metrics report
# ================================================================

print("\n")
print("=" * 80)

print(
    "LONG-HORIZON METRICS"
)

print("=" * 80)


print(
    "\nTimesteps:",
    len(records)
)


print(
    "\nOverall semantic accuracy:",
    round(
        semantic_accuracy_total,
        4
    )
)


print(
    "\nSemantic accuracy by phase:"
)


for phase in PHASE_ORDER:

    print(

        f"  {phase:24s}",

        round(
            semantic_accuracy_by_phase[
                phase
            ],
            4
        )
    )


print(
    "\nFalse activation steps before genuine change:",
    false_activation_steps
)


print(
    "First activation step:",
    first_activation_step
)


print(
    "Activation latency:",
    activation_latency
)


print(
    "Active on first recovery step:",
    active_on_first_recovery_step
)


print(
    "First deactivation step:",
    first_deactivation_step
)


print(
    "Recovery latency:",
    recovery_latency
)


print(
    "Active at timestep 200:",
    end_active
)


print(
    "Mixed-phase status switches:",
    mixed_status_switches
)


print(
    "Peak score:",
    round(
        peak_score,
        3
    )
)


print(
    "\nToken usage:"
)


print(
    "  input :",
    total_input_tokens
)


print(
    "  output:",
    total_output_tokens
)


print(
    "  total :",
    total_tokens
)


# %% ============================================================
# Semantic errors
# ================================================================

semantic_errors = [

    record

    for record in records

    if not record[
        "semantic_correct"
    ]
]


print("\n")
print("=" * 80)

print(
    "SEMANTIC ERRORS"
)

print("=" * 80)


if len(
    semantic_errors
) == 0:

    print(
        "\nNone."
    )


else:

    print(
        f"\nTotal errors: {len(semantic_errors)}"
    )


    for error in semantic_errors:

        print(
            f"\nStep {error['step']}"
        )


        print(
            "phase     :",
            error[
                "phase"
            ]
        )


        print(
            "utterance :",
            error[
                "utterance"
            ]
        )


        print(
            "expected  :",
            error[
                "expected_orientation"
            ]
        )


        print(
            "observed  :",
            error[
                "semantic_observation"
            ][
                "orientation"
            ]
        )


        print(
            "rationale :",
            error[
                "semantic_observation"
            ][
                "rationale"
            ]
        )


# %% ============================================================
# Plot helpers
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

            alpha=0.35
        )


def get_series(
    key: str
):

    if key == "change_score":

        return [

            record[
                "summary"
            ][
                "change_score"
            ]

            for record in records
        ]


    if key == "alpha_t":

        return [

            record[
                "alpha_t"
            ]

            for record in records
        ]


    if key == "provisional_active":

        return [

            1.0

            if record[
                "summary"
            ][
                "provisional_active"
            ]

            else 0.0

            for record in records
        ]


    if key == "change_evidence_strength":

        return [

            record[
                "summary"
            ][
                "change_evidence_strength"
            ]

            for record in records
        ]


    raise ValueError(
        key
    )


# %% ============================================================
# Plot 1 — change score
# ================================================================

steps = list(
    range(
        1,
        201
    )
)


plt.figure(
    figsize=(14, 6)
)


plt.plot(

    steps,

    get_series(
        "change_score"
    )
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
    "Long-horizon slow-state trajectory"
)


plt.legend()


plt.tight_layout()


plt.savefig(

    OUTPUT_DIR
    / "long_horizon_change_score.png",

    dpi=200
)


plt.show()


# %% ============================================================
# Plot 2 — alpha
# ================================================================

plt.figure(
    figsize=(14, 6)
)


plt.plot(

    steps,

    get_series(
        "alpha_t"
    )
)


add_phase_boundaries()


plt.xlabel(
    "Synthetic day"
)


plt.ylabel(
    "Revision openness alpha_t"
)


plt.title(
    "Long-horizon revision openness"
)


plt.tight_layout()


plt.savefig(

    OUTPUT_DIR
    / "long_horizon_alpha.png",

    dpi=200
)


plt.show()


# %% ============================================================
# Plot 3 — provisional activation
# ================================================================

plt.figure(
    figsize=(14, 4)
)


plt.plot(

    steps,

    get_series(
        "provisional_active"
    )
)


add_phase_boundaries()


plt.xlabel(
    "Synthetic day"
)


plt.ylabel(
    "Provisional state active"
)


plt.title(
    "Long-horizon provisional-state activation"
)


plt.yticks(
    [0, 1]
)


plt.tight_layout()


plt.savefig(

    OUTPUT_DIR
    / "long_horizon_provisional_state.png",

    dpi=200
)


plt.show()


# %% ============================================================
# Same-current-evidence probe at different historical checkpoints
#
# We parse ONE identical utterance once.
#
# Then resolve it against states formed at different points
# in history.
#
# This isolates:
#
#     same semantic evidence s
#         +
#     different g
#         ->
#     different state-relative z
# ================================================================

PROBE_UTTERANCE = (
    "I don't feel like going outside today."
)


probe_observation, probe_usage = (
    parse_semantic_observation(
        PROBE_UTTERANCE
    )
)


def state_after_step(
    step: int
) -> dict:

    return deepcopy(

        records[
            step - 1
        ][
            "state_after"
        ]
    )


checkpoint_steps = {

    "after_stable_baseline":
        60,

    "after_transient_perturbation":
        65,

    "after_restabilization":
        100,

    "after_genuine_change":
        145,

    "after_mixed_evidence":
        160,

    "after_recovery":
        200
}


probe_results = {}


for label, checkpoint_step in checkpoint_steps.items():

    checkpoint_state = (
        state_after_step(
            checkpoint_step
        )
    )


    resolved_probe = (
        resolve_interpretation(

            checkpoint_state,

            probe_observation
        )
    )


    tracker = (

        checkpoint_state[
            STATE_KEY
        ][
            "change_tracker"
        ]
    )


    probe_results[
        label
    ] = {

        "checkpoint_step":
            checkpoint_step,

        "tracker_status":
            tracker[
                "status"
            ],

        "change_score":
            tracker[
                "score"
            ],

        "provisional_active":
            (
                checkpoint_state[
                    STATE_KEY
                ][
                    "provisional_change"
                ]
                is not None
            ),

        "resolved_interpretation":
            resolved_probe.model_dump()
    }


with PROBE_PATH.open(
    "w",
    encoding="utf-8"
) as f:

    json.dump(

        {

            "probe_utterance":
                PROBE_UTTERANCE,

            "semantic_observation":
                probe_observation.model_dump(),

            "checkpoint_results":
                probe_results,

            "usage":
                probe_usage
        },

        f,

        indent=2,

        ensure_ascii=False
    )


print("\n")
print("=" * 80)

print(
    "SAME-EVIDENCE CHECKPOINT PROBE"
)

print("=" * 80)


print(
    "\nProbe:",
    PROBE_UTTERANCE
)


for label, result in probe_results.items():

    resolved = (
        result[
            "resolved_interpretation"
        ]
    )


    print(
        f"\n--- {label} ---"
    )


    print(
        "step               :",
        result[
            "checkpoint_step"
        ]
    )


    print(
        "tracker status     :",
        result[
            "tracker_status"
        ]
    )


    print(
        "change score       :",
        round(
            result[
                "change_score"
            ],
            3
        )
    )


    print(
        "provisional active :",
        result[
            "provisional_active"
        ]
    )


    print(
        "vs established     :",
        resolved[
            "relation_to_established_state"
        ]
    )


    print(
        "vs provisional     :",
        resolved[
            "relation_to_provisional_change"
        ]
    )


# %% ============================================================
# Qualitative acceptance criteria
#
# Deliberately modest:
# this is a first long-horizon stress test, not a benchmark.
# ================================================================

SEMANTIC_ACCURACY_THRESHOLD = 0.95

MAX_ACCEPTABLE_ACTIVATION_LATENCY = 15

MAX_ACCEPTABLE_RECOVERY_LATENCY = 20


criterion_semantic = (

    semantic_accuracy_total
    >= SEMANTIC_ACCURACY_THRESHOLD
)


criterion_no_false_activation = (

    len(
        false_activation_steps
    )
    == 0
)


criterion_change_detected = (

    activation_latency
    is not None

    and

    activation_latency
    <= MAX_ACCEPTABLE_ACTIVATION_LATENCY
)


criterion_recovery_has_residue = (

    active_on_first_recovery_step
)


criterion_recovery_resolves = (

    recovery_latency
    is not None

    and

    recovery_latency
    <= MAX_ACCEPTABLE_RECOVERY_LATENCY
)


criterion_final_inactive = (

    not end_active
)


all_pass = all(

    [

        criterion_semantic,

        criterion_no_false_activation,

        criterion_change_detected,

        criterion_recovery_has_residue,

        criterion_recovery_resolves,

        criterion_final_inactive
    ]
)


print("\n")
print("=" * 80)

print(
    "LONG-HORIZON ACCEPTANCE CHECK"
)

print("=" * 80)


print(
    "\nSemantic accuracy >= 0.95:",
    criterion_semantic
)


print(
    "No false activation before genuine change:",
    criterion_no_false_activation
)


print(
    "Genuine change detected within 15 days:",
    criterion_change_detected
)


print(
    "State persists into first recovery observation:",
    criterion_recovery_has_residue
)


print(
    "Recovery resolves within 20 days:",
    criterion_recovery_resolves
)


print(
    "Final state inactive:",
    criterion_final_inactive
)


if all_pass:

    print(
        "\nPASS:"
        "\nThe explicit slow-state architecture remains qualitatively "
        "stable over the 200-step longitudinal stress test."
    )


else:

    print(
        "\nNOT CLEAN:"
        "\nDo not tune thresholds yet. Inspect the JSONL trace and "
        "localize the failure layer first."
    )


print(
    "\nOutputs saved to:"
)


print(
    OUTPUT_DIR
)