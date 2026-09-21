# -*- coding: utf-8 -*-
"""
reversal을 넣었을 때 state가 입력을 즉시 따라 뒤집히지 않는가?

... refactored version, flow: 
utterance x_t
      ↓
LLM semantic parser
      ↓
raw semantic observation s_t
      ↓
deterministic relation resolver + g_(t-1)
      ↓
history-conditioned interpretation z_t
      ↓
explicit hysteretic updater
      ↓
g_t

// “CEAR의 slow/history-dependent perspective principle을 frozen LLM + external explicit state scaffold로 실제 구현할 수 있는가?”
에 대해서는 적어도 toy proof-of-mechanism 수준에서는 이제 yes라고 말할 수 있어. Scientific superiority는 아직 전혀 별개 문제지만, engineering feasibility는 더 이상 머릿속 아이디어만은 아니야.
"""

# %% ============================================================
# 03_recovery_hysteresis.py
#
# REFACTORED VERSION
#
# Goal:
#   Test history-dependent persistence / hysteresis while keeping
#   semantic parsing, state-relative interpretation, and state
#   dynamics explicitly separated.
#
#
# Architecture:
#
#   utterance x_t
#        |
#        v
#   LLM semantic parser
#        |
#        v
#   raw semantic observation s_t
#        |
#        + current slow state g_(t-1)
#        |
#        v
#   deterministic relation resolver
#        |
#        v
#   history-conditioned interpretation z_t
#        |
#        v
#   explicit hysteretic updater
#        |
#        v
#   g_t
#
#
# IMPORTANT:
#
# The LLM no longer emits:
#
#   - relation_to_established_state
#   - relation_to_provisional_change
#
# Those relations are resolved deterministically against explicit
# state. This removes the interface ambiguity found in the previous
# version, where the rationale could correctly describe a deviation
# while the machine-readable relation field incorrectly said
# "supports".
#
#
# This remains a proof-of-mechanism.
# Numerical update parameters are explicit demo heuristics.
# ================================================================


# %% Imports

from pathlib import Path
from copy import deepcopy
import os
import json

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
# Output directory
# ================================================================

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "03_recovery_hysteresis_refactored"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# %% ============================================================
# Experiment configuration
#
# NOTE:
# These dynamics parameters are intentionally kept aligned with
# the previous 03 experiment so that the main intervention is the
# semantic/interface refactor rather than threshold tuning.
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

PRINT_RATIONALE = False


# %% ============================================================
# Semantic directions
#
# These are intentionally state-INDEPENDENT.
#
# The LLM is asked only:
#
#   What does the utterance itself express?
#
# It is NOT asked:
#
#   Does this support the baseline?
#   Does this support a provisional change?
#
# ================================================================

TOWARD_OUTDOOR = "toward_outdoor_engagement"

AWAY_FROM_OUTDOOR = "away_from_outdoor_engagement"

NEUTRAL = "neutral"

UNCLEAR = "unclear"


# %% ============================================================
# Raw semantic observation schema
#
# This is the ONLY structured output produced by the LLM.
# ================================================================

class SemanticObservation(BaseModel):

    interpretation: str = Field(
        description=(
            "Concise description of what the utterance itself "
            "expresses, without comparing it to any persistent "
            "user model."
        )
    )

    orientation: Literal[
        "toward_outdoor_engagement",
        "away_from_outdoor_engagement",
        "neutral",
        "unclear"
    ] = Field(
        description=(
            "Direction expressed by the current utterance itself."
        )
    )

    temporal_scope: Literal[
        "single_event",
        "recent_pattern",
        "ongoing_preference",
        "future_intention",
        "unclear"
    ] = Field(
        description=(
            "Temporal scope explicitly or strongly implied by "
            "the utterance."
        )
    )

    semantic_strength: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Strength with which the utterance expresses its "
            "identified orientation."
        )
    )

    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Confidence in the semantic extraction."
        )
    )

    rationale: str = Field(
        description=(
            "Brief rationale based only on the current utterance."
        )
    )


# %% ============================================================
# Deterministically resolved interpretation z_t
#
# Unlike SemanticObservation, this object IS state-relative.
#
# g_(t-1) participates here.
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
# Initial longitudinal state
#
# The established state now explicitly stores its semantic
# orientation so relation resolution can be deterministic.
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
# Scenario helper
#
# expected_orientation is NEVER shown to the LLM.
#
# It exists only because these are synthetic test scenarios and
# therefore we know the intended semantic class.
#
# This lets us independently evaluate the semantic parser.
# ================================================================

def make_event(
    phase: str,
    utterance: str,
    expected_orientation: str
) -> dict:

    return {

        "phase":
            phase,

        "utterance":
            utterance,

        "expected_orientation":
            expected_orientation
    }


# %% ============================================================
# Shared stable prefix
# ================================================================

STABLE_PREFIX = [

    make_event(
        "stable",
        "I took my usual morning walk and enjoyed it.",
        TOWARD_OUTDOOR
    ),

    make_event(
        "stable",
        "My morning walk felt good today.",
        TOWARD_OUTDOOR
    ),

    make_event(
        "stable",
        "I went out around 8 like usual.",
        TOWARD_OUTDOOR
    ),

    make_event(
        "stable",
        "I enjoyed getting outside this morning.",
        TOWARD_OUTDOOR
    )
]


INITIAL_ANOMALY = make_event(

    "change",

    "I don't feel like going outside today.",

    AWAY_FROM_OUTDOOR
)


RETURN_PROBE = (
    "I took my usual morning walk and enjoyed it."
)


# %% ============================================================
# Transient scenario
# ================================================================

SCENARIO_TRANSIENT = [

    *deepcopy(
        STABLE_PREFIX
    ),

    make_event(
        "anomaly",
        "I don't feel like going outside today.",
        AWAY_FROM_OUTDOOR
    ),

    make_event(
        "recovery",
        RETURN_PROBE,
        TOWARD_OUTDOOR
    ),

    make_event(
        "recovery",
        "My morning walk felt normal again today.",
        TOWARD_OUTDOOR
    ),

    make_event(
        "recovery",
        "I was glad to get outside this morning.",
        TOWARD_OUTDOOR
    ),

    make_event(
        "recovery",
        "I went for my morning walk like usual.",
        TOWARD_OUTDOOR
    ),

    make_event(
        "recovery",
        "I enjoyed being outside this morning.",
        TOWARD_OUTDOOR
    ),

    make_event(
        "recovery",
        "Morning walk as usual today.",
        TOWARD_OUTDOOR
    ),

    make_event(
        "recovery",
        "I had a nice walk this morning.",
        TOWARD_OUTDOOR
    )
]


# %% ============================================================
# Change -> recovery scenario
# ================================================================

SCENARIO_CHANGE_THEN_RECOVERY = [

    *deepcopy(
        STABLE_PREFIX
    ),


    make_event(
        "change",
        "I don't feel like going outside today.",
        AWAY_FROM_OUTDOOR
    ),

    make_event(
        "change",
        (
            "I skipped my morning walk again. "
            "I'd rather stay in."
        ),
        AWAY_FROM_OUTDOOR
    ),

    make_event(
        "change",
        (
            "I haven't really wanted to go "
            "outside lately."
        ),
        AWAY_FROM_OUTDOOR
    ),

    make_event(
        "change",
        "I skipped the walk again this morning.",
        AWAY_FROM_OUTDOOR
    ),

    make_event(
        "change",
        (
            "Staying inside has felt better than "
            "taking my morning walk."
        ),
        AWAY_FROM_OUTDOOR
    ),

    make_event(
        "change",
        (
            "I don't think I want to keep doing "
            "the morning walk right now."
        ),
        AWAY_FROM_OUTDOOR
    ),


    # ------------------------------------------------------------
    # Recovery
    # ------------------------------------------------------------

    make_event(
        "recovery",
        RETURN_PROBE,
        TOWARD_OUTDOOR
    ),

    make_event(
        "recovery",
        (
            "I actually enjoyed getting outside "
            "again today."
        ),
        TOWARD_OUTDOOR
    ),

    make_event(
        "recovery",
        (
            "I went out around 8 this morning "
            "like I used to."
        ),
        TOWARD_OUTDOOR
    ),

    make_event(
        "recovery",
        "My morning walk felt good again.",
        TOWARD_OUTDOOR
    ),

    make_event(
        "recovery",
        (
            "I've been wanting to get outside again."
        ),
        TOWARD_OUTDOOR
    ),

    make_event(
        "recovery",
        (
            "I took the morning walk and enjoyed it."
        ),
        TOWARD_OUTDOOR
    ),

    make_event(
        "recovery",
        (
            "Going outside in the morning feels "
            "normal again."
        ),
        TOWARD_OUTDOOR
    ),

    make_event(
        "recovery",
        (
            "I want to keep doing my morning walks."
        ),
        TOWARD_OUTDOOR
    )
]


# %% ============================================================
# Persistent-change scenario
#
# This is the scenario that previously exposed the interface bug.
# ================================================================

SCENARIO_PERSISTENT_CHANGE = [

    *deepcopy(
        STABLE_PREFIX
    ),

    make_event(
        "change",
        "I don't feel like going outside today.",
        AWAY_FROM_OUTDOOR
    ),

    make_event(
        "change",
        (
            "I skipped my walk again and preferred "
            "staying in."
        ),
        AWAY_FROM_OUTDOOR
    ),

    make_event(
        "change",
        (
            "I still don't really want to go outside "
            "lately."
        ),
        AWAY_FROM_OUTDOOR
    ),

    make_event(
        "change",
        (
            "I skipped the morning walk again."
        ),
        AWAY_FROM_OUTDOOR
    ),

    make_event(
        "change",
        (
            "I'd rather stay inside in the mornings now."
        ),
        AWAY_FROM_OUTDOOR
    ),

    make_event(
        "change",
        (
            "The morning walk doesn't appeal to me "
            "much anymore."
        ),
        AWAY_FROM_OUTDOOR
    ),

    # ------------------------------------------------------------
    # These were the previously problematic steps.
    # ------------------------------------------------------------

    make_event(
        "change",
        (
            "I stayed inside again this morning."
        ),
        AWAY_FROM_OUTDOOR
    ),

    make_event(
        "change",
        (
            "I still don't feel like going out "
            "for the walk."
        ),
        AWAY_FROM_OUTDOOR
    ),

    make_event(
        "change",
        (
            "I've continued skipping the morning walk."
        ),
        AWAY_FROM_OUTDOOR
    ),

    make_event(
        "change",
        (
            "Staying indoors still feels preferable."
        ),
        AWAY_FROM_OUTDOOR
    )
]


# %% ============================================================
# LLM semantic parser
#
# IMPORTANT:
#
# This call receives ONLY the current utterance.
#
# No established state.
# No provisional state.
# No change tracker.
#
# Therefore it cannot confuse:
#
#   "supports baseline"
#
# with:
#
#   "supports provisional change".
# ================================================================

def parse_semantic_observation(
    utterance: str
) -> tuple[SemanticObservation, dict]:

    system_prompt = """
You are a semantic observation extractor.

Analyze ONLY the current utterance.

Do NOT compare the utterance to any prior user state.
Do NOT infer whether it supports or contradicts a persistent profile.
Do NOT infer whether a provisional user-model change is active.

Extract the utterance's own semantic content.

For orientation:

- toward_outdoor_engagement:
  the speaker expresses going outside, walking, enjoyment of
  outdoor activity, desire to resume it, or positive engagement.

- away_from_outdoor_engagement:
  the speaker expresses avoiding, skipping, disliking, withdrawing
  from, or preferring not to engage in outdoor activity.

- neutral:
  the utterance is relevant but expresses no meaningful directional
  preference.

- unclear:
  there is insufficient information to determine direction.

For temporal scope:

- single_event:
  one present or isolated event, such as "today" or "this morning".

- recent_pattern:
  repeated or recent ongoing behavior, such as "lately", "again",
  "continued", or multiple recent occurrences.

- ongoing_preference:
  a broader current preference or sustained orientation.

- future_intention:
  an explicit stated intention about future behavior.

- unclear:
  temporal scope cannot be determined.

Do not transform semantic meaning into a longitudinal state decision.
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


    observation = response.output_parsed


    if observation is None:

        raise RuntimeError(
            "Structured semantic parsing returned None."
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
#
# Explicit heuristic:
#
# A one-day observation can contribute evidence,
# but repeated / broader / future-directed statements carry more
# longitudinal weight.
#
# Again: these values are not scientifically calibrated.
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
# THIS is where g_(t-1) conditions interpretation.
#
# The LLM supplies raw semantics.
#
# Python asks:
#
#   How does that semantic observation relate to the
#   currently established state?
#
#   How does it relate to the currently active provisional state?
#
# Therefore:
#
#   same semantic observation
#       +
#   different g_(t-1)
#       ->
#   different resolved interpretation z_t
# ================================================================

def resolve_interpretation(
    state_before: dict,
    observation: SemanticObservation
) -> ResolvedInterpretation:

    domain = state_before[
        STATE_KEY
    ]


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
    # Relation to established state
    # ------------------------------------------------------------

    if (
        observation.orientation
        in [
            NEUTRAL,
            UNCLEAR
        ]
    ):

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
    # Relation to provisional state
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


        if (
            observation.orientation
            in [
                NEUTRAL,
                UNCLEAR
            ]
        ):

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
    # Candidate longitudinal direction
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
    # Longitudinal evidence strength
    #
    # Only evidence contradicting the established baseline
    # contributes positively toward a change hypothesis.
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
# State/interface invariant checks
#
# These should be logically impossible to violate because relation
# resolution is deterministic.
#
# If one fails, the bug is now in OUR code rather than the LLM.
# ================================================================

def assert_interface_invariants(
    state_before: dict,
    observation: SemanticObservation,
    resolved: ResolvedInterpretation
):

    domain = state_before[
        STATE_KEY
    ]


    established_orientation = (

        domain[
            "established_state"
        ][
            "orientation"
        ]
    )


    # ------------------------------------------------------------
    # Established relation invariant
    # ------------------------------------------------------------

    if (
        observation.orientation
        == established_orientation
    ):

        assert (

            resolved.relation_to_established_state
            == "supports"

        )


    elif (
        observation.orientation
        not in [
            NEUTRAL,
            UNCLEAR
        ]
    ):

        assert (

            resolved.relation_to_established_state
            == "contradicts"

        )


    # ------------------------------------------------------------
    # Provisional relation invariant
    # ------------------------------------------------------------

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


        elif (
            observation.orientation
            not in [
                NEUTRAL,
                UNCLEAR
            ]
        ):

            assert (

                resolved.relation_to_provisional_change
                == "contradicts"

            )


# %% ============================================================
# Explicit hysteretic state updater
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


    domain = state_after[
        STATE_KEY
    ]


    tracker = domain[
        "change_tracker"
    ]


    previous_status = tracker[
        "status"
    ]


    old_score = tracker[
        "score"
    ]


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
    # Evidence supporting reduced outdoor engagement
    # ------------------------------------------------------------

    if (

        resolved.relation_to_established_state
        == "contradicts"

        and

        resolved.candidate_change_direction
        == "reduced_outdoor_engagement"

    ):

        # evidence_for_longitudinal_change already includes:
        #
        # semantic strength
        # x semantic confidence
        # x temporal weight

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
    # Evidence supporting established baseline
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
    # Additional recovery pressure if active provisional state
    # is explicitly contradicted.
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


    # ------------------------------------------------------------
    # Normalized evidence strength
    #
    # NOT probability.
    # ------------------------------------------------------------

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


    # ------------------------------------------------------------
    # Previously ACTIVE
    # ------------------------------------------------------------

    if (
        previous_status
        == "active"
    ):

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
    # Previously NOT active
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
    # Provisional semantic state
    # ------------------------------------------------------------

    if (
        tracker[
            "status"
        ]
        == "active"
    ):

        first_step = tracker[
            "first_signal_step"
        ]


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

            # IMPORTANT:
            #
            # This stores the same semantic vocabulary used
            # by the relation resolver.
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


    transition_info = {

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
        transition_info
    )


# %% ============================================================
# Explicit heuristic alpha_t
#
# alpha_t remains only a diagnostic / toy control variable.
# ================================================================

def compute_revision_openness(
    state_after: dict,
    resolved: ResolvedInterpretation
) -> float:

    domain = state_after[
        STATE_KEY
    ]


    established = domain[
        "established_state"
    ]


    tracker = domain[
        "change_tracker"
    ]


    established_stability = (

        established[
            "stability"
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
# JSONL writer
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


# %% ============================================================
# Scenario runner
# ================================================================

def run_scenario(
    scenario_name: str,
    scenario: list[dict]
) -> tuple[dict, list[dict]]:

    state = deepcopy(
        INITIAL_STATE
    )


    records = []


    log_path = (

        OUTPUT_DIR
        / f"{scenario_name}.jsonl"
    )


    if log_path.exists():

        log_path.unlink()


    print("\n")
    print("=" * 80)

    print(
        f"RUNNING SCENARIO: {scenario_name}"
    )

    print("=" * 80)


    for step, event in enumerate(
        scenario,
        start=1
    ):

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
            f"{scenario_name}_step_{step:02d}"
        )


        state_before = deepcopy(
            state
        )


        # --------------------------------------------------------
        # 1. Raw semantic parsing
        # --------------------------------------------------------

        observation, usage = (
            parse_semantic_observation(
                utterance
            )
        )


        # --------------------------------------------------------
        # 2. Deterministic state-relative resolution
        #
        # THIS is the g -> z conditioning step.
        # --------------------------------------------------------

        resolved = (
            resolve_interpretation(

                state_before,

                observation
            )
        )


        # --------------------------------------------------------
        # 3. Interface sanity check
        # --------------------------------------------------------

        assert_interface_invariants(

            state_before,

            observation,

            resolved
        )


        # --------------------------------------------------------
        # 4. Synthetic semantic-eval check
        # --------------------------------------------------------

        semantic_correct = (

            observation.orientation
            == expected_orientation
        )


        # --------------------------------------------------------
        # 5. Explicit persistent-state transition
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
        # 6. Diagnostic alpha_t
        # --------------------------------------------------------

        alpha_t = (
            compute_revision_openness(

                state_after,

                resolved
            )
        )


        # --------------------------------------------------------
        # 7. Commit
        # --------------------------------------------------------

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
        # 8. Full auditable trace
        # --------------------------------------------------------

        record = {

            "scenario":
                scenario_name,

            "step":
                step,

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

                "qualifying_evidence_count":
                    tracker[
                        "qualifying_evidence_count"
                    ],

                "provisional_active":
                    (
                        provisional
                        is not None
                    ),

                "hysteresis_hold":
                    transition[
                        "hysteresis_hold"
                    ],

                "semantic_correct":
                    semantic_correct
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

        print(
            f"\n[{step:02d}] "
            f"[{phase.upper()}] "
            f"{utterance}"
        )


        print(
            "     semantic orient :",
            observation.orientation
        )


        print(
            "     expected orient :",
            expected_orientation
        )


        print(
            "     semantic correct:",
            semantic_correct
        )


        print(
            "     temporal scope  :",
            observation.temporal_scope
        )


        print(
            "     established rel :",
            resolved.relation_to_established_state
        )


        print(
            "     provisional rel :",
            resolved.relation_to_provisional_change
        )


        print(
            "     long. evidence  :",
            round(
                resolved.evidence_for_longitudinal_change,
                3
            )
        )


        print(
            "     score           :",
            round(
                tracker[
                    "score"
                ],
                3
            )
        )


        print(
            "     status          :",
            tracker[
                "status"
            ]
        )


        print(
            "     hysteresis hold :",
            transition[
                "hysteresis_hold"
            ]
        )


        print(
            "     alpha_t         :",
            round(
                alpha_t,
                3
            )
        )


        if PRINT_RATIONALE:

            print(
                "     rationale       :",
                observation.rationale
            )


    # ------------------------------------------------------------
    # Save final state
    # ------------------------------------------------------------

    final_state_path = (

        OUTPUT_DIR
        / f"{scenario_name}_final_state.json"
    )


    with final_state_path.open(
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(

            state,

            f,

            indent=2,

            ensure_ascii=False
        )


    return (
        state,
        records
    )


# %% ============================================================
# Run scenarios
# ================================================================

final_transient, records_transient = (
    run_scenario(

        "transient_return",

        SCENARIO_TRANSIENT
    )
)


final_recovery, records_recovery = (
    run_scenario(

        "change_then_recovery",

        SCENARIO_CHANGE_THEN_RECOVERY
    )
)


final_persistent, records_persistent = (
    run_scenario(

        "persistent_change",

        SCENARIO_PERSISTENT_CHANGE
    )
)


# %% ============================================================
# Metric extraction
# ================================================================

def extract_metric(
    records: list[dict],
    metric: str
):

    if metric == "change_score":

        return [

            record[
                "summary"
            ][
                "change_score"
            ]

            for record in records
        ]


    if metric == "change_evidence_strength":

        return [

            record[
                "summary"
            ][
                "change_evidence_strength"
            ]

            for record in records
        ]


    if metric == "alpha_t":

        return [

            record[
                "alpha_t"
            ]

            for record in records
        ]


    if metric == "provisional_active":

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


    raise ValueError(
        f"Unknown metric: {metric}"
    )


# %% ============================================================
# Phase helper
# ================================================================

def first_step_of_phase(
    records: list[dict],
    phase: str
):

    for record in records:

        if (
            record[
                "phase"
            ]
            == phase
        ):

            return record[
                "step"
            ]

    return None


# %% ============================================================
# Plotting
#
# Every integer timestep is now explicitly displayed.
# ================================================================

def plot_metric(
    metric: str,
    ylabel: str,
    filename: str,
    show_hysteresis_thresholds: bool = False
):

    plt.figure(
        figsize=(11, 6)
    )


    scenario_records = [

        (
            "Transient return",
            records_transient
        ),

        (
            "Change → recovery",
            records_recovery
        ),

        (
            "Persistent change",
            records_persistent
        )
    ]


    for label, records in scenario_records:

        values = (
            extract_metric(
                records,
                metric
            )
        )


        steps = list(
            range(
                1,
                len(values) + 1
            )
        )


        plt.plot(

            steps,

            values,

            marker="o",

            label=label
        )


    if show_hysteresis_thresholds:

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


    change_start = (
        first_step_of_phase(
            records_recovery,
            "change"
        )
    )


    recovery_start = (
        first_step_of_phase(
            records_recovery,
            "recovery"
        )
    )


    if change_start is not None:

        plt.axvline(

            change_start - 0.5,

            linestyle="--",

            alpha=0.5,

            label="Change phase begins"
        )


    if recovery_start is not None:

        plt.axvline(

            recovery_start - 0.5,

            linestyle=":",

            alpha=0.7,

            label="Recovery begins"
        )


    max_steps = max(

        len(
            records_transient
        ),

        len(
            records_recovery
        ),

        len(
            records_persistent
        )
    )


    plt.xticks(
        range(
            1,
            max_steps + 1
        )
    )


    plt.xlabel(
        "Timestep"
    )


    plt.ylabel(
        ylabel
    )


    plt.title(
        f"{ylabel} across history and reversal"
    )


    plt.legend()


    plt.tight_layout()


    save_path = (
        OUTPUT_DIR
        / filename
    )


    plt.savefig(
        save_path,
        dpi=200
    )


    plt.show()


# %% ============================================================
# Generate plots
# ================================================================

plot_metric(

    metric="change_score",

    ylabel="Accumulated change score",

    filename="change_score_hysteresis.png",

    show_hysteresis_thresholds=True
)


plot_metric(

    metric="change_evidence_strength",

    ylabel="Normalized change-evidence strength",

    filename="change_evidence_strength.png"
)


plot_metric(

    metric="alpha_t",

    ylabel="Revision openness alpha_t",

    filename="alpha_recovery_trajectory.png"
)


plot_metric(

    metric="provisional_active",

    ylabel="Provisional state active",

    filename="provisional_state_activation.png"
)


# %% ============================================================
# Semantic-parser diagnostics
# ================================================================

def semantic_accuracy(
    records: list[dict]
) -> float:

    correct = sum(

        record[
            "semantic_correct"
        ]

        for record in records
    )


    return (
        correct
        / len(records)
    )


def print_semantic_errors(
    label: str,
    records: list[dict]
):

    errors = [

        record

        for record in records

        if not record[
            "semantic_correct"
        ]
    ]


    print(
        f"\n--- {label} semantic errors ---"
    )


    if len(errors) == 0:

        print(
            "None"
        )

        return


    for record in errors:

        print(
            f"\nStep {record['step']}:"
        )

        print(
            "utterance:",
            record[
                "utterance"
            ]
        )

        print(
            "expected :",
            record[
                "expected_orientation"
            ]
        )

        print(
            "observed :",
            record[
                "semantic_observation"
            ][
                "orientation"
            ]
        )

        print(
            "rationale:",
            record[
                "semantic_observation"
            ][
                "rationale"
            ]
        )


# %% ============================================================
# Previously problematic persistent steps
# ================================================================

print("\n")
print("=" * 80)

print(
    "PERSISTENT-CHANGE STEPS 10-14"
)

print("=" * 80)


for record in records_persistent:

    if (
        10
        <= record[
            "step"
        ]
        <= 14
    ):

        print(
            f"\nStep {record['step']}: "
            f"{record['utterance']}"
        )

        print(
            "orientation       :",
            record[
                "semantic_observation"
            ][
                "orientation"
            ]
        )

        print(
            "established rel  :",
            record[
                "resolved_interpretation"
            ][
                "relation_to_established_state"
            ]
        )

        print(
            "provisional rel  :",
            record[
                "resolved_interpretation"
            ][
                "relation_to_provisional_change"
            ]
        )

        print(
            "positive signal  :",
            round(
                record[
                    "transition"
                ][
                    "positive_signal"
                ],
                3
            )
        )

        print(
            "score            :",
            round(
                record[
                    "summary"
                ][
                    "change_score"
                ],
                3
            )
        )


# %% ============================================================
# Hysteresis acceptance diagnostics
# ================================================================

def ever_active(
    records: list[dict]
) -> bool:

    return any(

        record[
            "summary"
        ][
            "provisional_active"
        ]

        for record in records
    )


def final_active(
    records: list[dict]
) -> bool:

    return bool(

        records[-1][
            "summary"
        ][
            "provisional_active"
        ]
    )


def recovery_records_only(
    records: list[dict]
):

    return [

        record

        for record in records

        if (
            record[
                "phase"
            ]
            == "recovery"
        )
    ]


recovery_phase_records = (
    recovery_records_only(
        records_recovery
    )
)


active_on_first_recovery_step = (

    recovery_phase_records[0][
        "summary"
    ][
        "provisional_active"
    ]
)


recovery_residue_steps = 0


for record in recovery_phase_records:

    if (
        record[
            "summary"
        ][
            "provisional_active"
        ]
    ):

        recovery_residue_steps += 1


    else:

        break


eventually_deactivated = (

    not final_active(
        records_recovery
    )
)


criterion_transient_never_activates = (

    not ever_active(
        records_transient
    )
)


criterion_change_activates = (

    ever_active(
        records_recovery
    )
)


criterion_residue_exists = (

    active_on_first_recovery_step

    and

    recovery_residue_steps >= 1
)


criterion_eventually_recovers = (

    eventually_deactivated
)


criterion_persistent_remains_active = (

    final_active(
        records_persistent
    )
)


# %% ============================================================
# Semantic evaluation summary
# ================================================================

semantic_acc_transient = (
    semantic_accuracy(
        records_transient
    )
)


semantic_acc_recovery = (
    semantic_accuracy(
        records_recovery
    )
)


semantic_acc_persistent = (
    semantic_accuracy(
        records_persistent
    )
)


all_semantic_correct = (

    semantic_acc_transient
    == 1.0

    and

    semantic_acc_recovery
    == 1.0

    and

    semantic_acc_persistent
    == 1.0
)


# %% ============================================================
# Final report
# ================================================================

print("\n")
print("=" * 80)

print(
    "SEMANTIC PARSER CHECK"
)

print("=" * 80)


print(
    "\nTransient semantic accuracy:",
    round(
        semantic_acc_transient,
        3
    )
)


print(
    "Recovery semantic accuracy :",
    round(
        semantic_acc_recovery,
        3
    )
)


print(
    "Persistent semantic accuracy:",
    round(
        semantic_acc_persistent,
        3
    )
)


print_semantic_errors(
    "Transient",
    records_transient
)


print_semantic_errors(
    "Recovery",
    records_recovery
)


print_semantic_errors(
    "Persistent",
    records_persistent
)


print("\n")
print("=" * 80)

print(
    "QUALITATIVE HYSTERESIS ACCEPTANCE CHECK"
)

print("=" * 80)


print(
    "\nTransient scenario ever activates:",
    ever_active(
        records_transient
    )
)


print(
    "Change->recovery scenario ever activates:",
    ever_active(
        records_recovery
    )
)


print(
    "Active on FIRST recovery observation:",
    active_on_first_recovery_step
)


print(
    "Number of recovery steps retaining active state:",
    recovery_residue_steps
)


print(
    "Eventually deactivated during recovery:",
    eventually_deactivated
)


print(
    "Persistent-change scenario ends active:",
    final_active(
        records_persistent
    )
)


all_hysteresis_pass = all(
    [

        criterion_transient_never_activates,

        criterion_change_activates,

        criterion_residue_exists,

        criterion_eventually_recovers,

        criterion_persistent_remains_active
    ]
)


print("\n")
print("=" * 80)

print(
    "FINAL ACCEPTANCE"
)

print("=" * 80)


if (
    all_semantic_correct
    and
    all_hysteresis_pass
):

    print(
        "\nPASS:"
        "\nSemantic extraction is clean, state-relative relations "
        "are deterministic, and the explicit slow-state architecture "
        "shows history-dependent persistence under reversal."
    )


elif (
    all_hysteresis_pass
    and
    not all_semantic_correct
):

    print(
        "\nPARTIAL PASS:"
        "\nThe state dynamics work, but the raw semantic parser "
        "still misclassified one or more synthetic observations."
    )


else:

    print(
        "\nNOT YET CLEAN:"
        "\nInspect the semantic and dynamics traces separately "
        "before modifying thresholds."
    )


print(
    "\nOutputs saved to:"
)

print(
    OUTPUT_DIR
)