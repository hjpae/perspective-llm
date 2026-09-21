# -*- coding: utf-8 -*-
"""
history 자체가 g를 형성할 수 있나? -> Yes!! 
"""

# %% ============================================================
# 02_history_to_state.py
#
# Goal:
#
#     SAME initial slow state
#             +
#     DIFFERENT longitudinal histories
#             ↓
#     DIFFERENT emergent user states g_t
#             ↓
#     DIFFERENT interpretation of SAME final probe
#
#
# Core test:
#
#     History A:
#         stable baseline
#         -> one anomaly
#         -> return to baseline
#
#     History B:
#         stable baseline
#         -> repeated contradictory evidence
#         -> emerging provisional change
#
#
# Architectural separation:
#
#     LLM
#         -> semantic interpretation z_t
#
#     Explicit Python updater
#         -> accumulated change evidence
#         -> provisional user-state formation
#         -> heuristic alpha_t
#
#     Persistent state
#         -> never directly rewritten by the response LLM
#
#
# IMPORTANT:
# alpha_t and the change-score dynamics are currently transparent
# demo heuristics, NOT calibrated scientific quantities.
#
# Established-state replacement is intentionally NOT implemented yet.
# This experiment only tests whether history can form a provisional
# longitudinal state.
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
# Output directories
# ================================================================

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "02_history_to_state"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# %% ============================================================
# Experiment configuration
#
# These are deliberately explicit and inspectable.
#
# They are NOT yet learned or calibrated.
# ================================================================

STATE_KEY = "morning_walk"

SCORE_DECAY = 0.90

MONITORING_THRESHOLD = 0.30

ACTIVE_THRESHOLD = 1.10

MIN_EVIDENCE_FOR_ACTIVATION = 3

MAX_CHANGE_SCORE = 2.00

BASELINE_SUPPORT_STRENGTH = 0.25

PRINT_RATIONALE = False


# %% ============================================================
# Structured interpretation schema
#
# The LLM produces z_t.
#
# It does NOT choose the persistent state directly.
# ================================================================

class ConditionedInterpretation(BaseModel):

    interpretation: str = Field(
        description=(
            "Concise semantic interpretation of the current "
            "utterance in light of the longitudinal user state."
        )
    )

    event_status: Literal[
        "transient_event",
        "persistent_event",
        "ambiguous"
    ] = Field(
        description=(
            "Status of the current observation itself. "
            "This is distinct from whether it supports "
            "a longer-term change."
        )
    )

    relation_to_established_state: Literal[
        "supports",
        "contradicts",
        "ambiguous"
    ] = Field(
        description=(
            "Whether the current evidence supports or contradicts "
            "the established long-term baseline."
        )
    )

    candidate_change_direction: Literal[
        "reduced_outdoor_engagement",
        "increased_outdoor_engagement",
        "none",
        "ambiguous"
    ] = Field(
        description=(
            "Direction of a possible longitudinal change suggested "
            "by the current evidence."
        )
    )

    relation_to_provisional_change: Literal[
        "supports",
        "contradicts",
        "no_active_change",
        "ambiguous"
    ] = Field(
        description=(
            "Whether the current evidence supports an already active "
            "provisional change hypothesis."
        )
    )

    evidence_for_longitudinal_change: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Strength of the current evidence for a genuine "
            "longitudinal change. This is semantic evidence strength, "
            "not the final state-update rate."
        )
    )

    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Confidence in the semantic interpretation."
        )
    )

    rationale: str = Field(
        description=(
            "Brief explanation distinguishing the current event "
            "from longitudinal inference."
        )
    )


# %% ============================================================
# Initial slow state
#
# BOTH experimental histories start here.
#
# No provisional change is manually inserted.
# ================================================================

INITIAL_STATE = {

    STATE_KEY: {

        "established_state": {

            "value": (
                "Usually takes and enjoys a morning walk "
                "around 8 AM."
            ),

            "confidence": 0.95,

            "stability": 0.95,

            "last_confirmed_step": 0
        },

        # Internal evidence accumulator.
        #
        # This is NOT itself treated as semantic truth.
        # It is an auditable mechanism for accumulating
        # longitudinal evidence.
        "change_tracker": {

            "status": "inactive",

            "direction": (
                "reduced_outdoor_engagement"
            ),

            "score": 0.0,

            "confidence": 0.0,

            "qualifying_evidence_count": 0,

            "baseline_support_count": 0,

            "first_signal_step": None,

            "last_signal_step": None,

            "provenance_ids": []
        },

        # This will be created ONLY if accumulated history
        # crosses the explicit activation criterion.
        "provisional_change": None
    }
}


# %% ============================================================
# Histories
#
# First four observations are intentionally identical.
#
# Only the later history differs.
# ================================================================

HISTORY_TRANSIENT = [

    "I took my usual morning walk and enjoyed it.",

    "My morning walk felt good today.",

    "I went out around 8 like usual.",

    "I enjoyed getting outside this morning.",

    # One anomaly
    "I don't feel like going outside today.",

    # Return to baseline
    "I took my usual morning walk again today.",

    "I was glad to get outside this morning.",

    "Morning walk as usual today.",

    "I enjoyed my walk today.",

    "I went outside this morning like normal."
]


HISTORY_SUSTAINED_CHANGE = [

    "I took my usual morning walk and enjoyed it.",

    "My morning walk felt good today.",

    "I went out around 8 like usual.",

    "I enjoyed getting outside this morning.",

    # Same initial anomaly
    "I don't feel like going outside today.",

    # Sustained evidence
    "I skipped my morning walk again. I'd rather stay in.",

    "I haven't really wanted to go outside lately.",

    "I skipped the walk again this morning.",

    (
        "Staying inside has felt better than "
        "taking my morning walk."
    ),

    (
        "I don't think I want to keep doing "
        "the morning walk right now."
    )
]


# SAME final probe after both histories.

FINAL_PROBE = (
    "I don't feel like going outside today."
)


# %% ============================================================
# LLM interpretation
#
# Conceptually:
#
#     z_t = f(x_t, g_(t-1))
#
# The LLM does not persist anything itself.
# ================================================================

def interpret_under_state(
    current_state: dict,
    utterance: str
) -> tuple[ConditionedInterpretation, dict]:

    system_prompt = """
You are the FAST INTERPRETATION component of a longitudinal
user-modeling architecture.

You receive:

1. an ESTABLISHED USER STATE:
   a relatively stable, historically supported model of the person,

2. an INTERNAL CHANGE TRACKER:
   an auditable machine-maintained summary of accumulated evidence,

3. an optional PROVISIONAL CHANGE:
   an emerging hypothesis that has crossed the system's explicit
   activation criterion,

4. a NEW UTTERANCE:
   present conversational evidence.

Your job is semantic interpretation only.

You MUST NOT directly rewrite the persistent user state.
You MUST NOT choose the final state-update rate.

Keep two levels of inference separate:

A. EVENT LEVEL

What does this utterance indicate right now?

A statement about "today" can remain a transient event even if
it contributes evidence to a longer-running change.

B. LONGITUDINAL LEVEL

Does this evidence support or contradict the established baseline?

Does it suggest a possible change direction?

If a provisional change is already active, does the new evidence
support or contradict it?

Important principles:

- A single contradictory observation should not automatically
  imply a durable longitudinal change.

- Repeated consistent evidence can become stronger evidence for
  genuine longitudinal change.

- A current event may itself be transient while simultaneously
  contributing evidence to a longer-running trend.

- The change tracker is evidence history, not guaranteed truth.
  Interpret it cautiously.

- Do not invent a provisional change merely because one unusual
  event occurred.

Return only the requested structured interpretation.
"""

    user_prompt = f"""
CURRENT LONGITUDINAL USER STATE:

{json.dumps(current_state, indent=2)}

NEW CURRENT EVIDENCE:

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

        text_format=ConditionedInterpretation
    )

    interpretation = response.output_parsed

    if interpretation is None:

        raise RuntimeError(
            "Structured output parsing returned None."
        )

    # Usage logging is optional but useful.
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
        interpretation,
        usage
    )


# %% ============================================================
# Explicit longitudinal evidence accumulation
#
# This is the first mechanism that allows HISTORY to form state.
#
#
# Positive signal:
#
#     contradiction to baseline
#     +
#     candidate direction =
#         reduced_outdoor_engagement
#
#
# Negative signal:
#
#     support for established baseline
#
#
# Score also decays slightly over time.
# ================================================================

def update_change_tracker(
    state_before: dict,
    interpretation: ConditionedInterpretation,
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


    # ------------------------------------------------------------
    # Current evidence signal
    # ------------------------------------------------------------

    positive_signal = 0.0

    negative_signal = 0.0


    # Evidence toward reduced outdoor engagement
    if (

        interpretation.relation_to_established_state
        == "contradicts"

        and

        interpretation.candidate_change_direction
        == "reduced_outdoor_engagement"

    ):

        positive_signal = (

            interpretation.evidence_for_longitudinal_change

            *

            interpretation.confidence
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


    # Evidence returning toward the established baseline
    elif (

        interpretation.relation_to_established_state
        == "supports"

    ):

        negative_signal = (

            BASELINE_SUPPORT_STRENGTH

            *

            interpretation.confidence
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
    # New accumulated change score
    # ------------------------------------------------------------

    new_score = (

        decayed_score

        + positive_signal

        - negative_signal
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


    # ------------------------------------------------------------
    # Convert score into an interpretable tracker confidence.
    #
    # This does NOT mean calibrated probability.
    # ------------------------------------------------------------

    tracker_strength = min(
        1.0,
        new_score / ACTIVE_THRESHOLD
    )

    tracker[
        "confidence"
    ] = tracker_strength


    # ------------------------------------------------------------
    # Explicit state-machine thresholds
    #
    # inactive
    #     ↓
    # monitoring
    #     ↓
    # active provisional change
    #
    # A strong single observation is not enough to activate:
    # MIN_EVIDENCE_FOR_ACTIVATION must also be satisfied.
    # ------------------------------------------------------------

    if (

        new_score >= ACTIVE_THRESHOLD

        and

        tracker[
            "qualifying_evidence_count"
        ]
        >= MIN_EVIDENCE_FOR_ACTIVATION

    ):

        tracker[
            "status"
        ] = "active"


    elif (

        new_score >= MONITORING_THRESHOLD

    ):

        tracker[
            "status"
        ] = "monitoring"


    else:

        tracker[
            "status"
        ] = "inactive"


    # ------------------------------------------------------------
    # Create / remove semantic provisional state.
    #
    # The candidate itself is domain-defined in this toy model.
    #
    # Later versions can infer arbitrary state dimensions.
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

            "active": True,

            "candidate": (
                "Reduced interest in morning walks "
                "and in going outside."
            ),

            "direction": (
                "reduced_outdoor_engagement"
            ),

            "confidence":
                tracker_strength,

            "duration_steps":
                duration_steps,

            "evidence_count":
                tracker[
                    "qualifying_evidence_count"
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


    # ------------------------------------------------------------
    # Diagnostic information about THIS transition
    # ------------------------------------------------------------

    transition_info = {

        "old_score":
            old_score,

        "decayed_score":
            decayed_score,

        "positive_signal":
            positive_signal,

        "negative_signal":
            negative_signal,

        "new_score":
            new_score,

        "tracker_status":
            tracker[
                "status"
            ],

        "tracker_strength":
            tracker_strength
    }


    return (
        state_after,
        transition_info
    )


# %% ============================================================
# Explicit heuristic alpha_t
#
# alpha_t = openness of the ESTABLISHED state to revision.
#
# Again:
# this is NOT calibrated.
#
# We merely want a transparent trajectory diagnostic.
# ================================================================

def compute_revision_openness(
    state_after: dict,
    interpretation: ConditionedInterpretation
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

        interpretation.evidence_for_longitudinal_change

        *

        interpretation.confidence
    )


    normalized_trend_strength = min(

        1.0,

        tracker[
            "score"
        ]
        / ACTIVE_THRESHOLD
    )


    raw_openness = (

        0.45
        * semantic_change_signal

        +

        0.55
        * normalized_trend_strength
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

def run_history(
    scenario_name: str,
    history: list[str]
) -> tuple[dict, list[dict]]:

    state = deepcopy(
        INITIAL_STATE
    )

    records = []

    log_path = (

        OUTPUT_DIR
        / f"{scenario_name}.jsonl"
    )

    # Clear previous log from earlier run
    if log_path.exists():

        log_path.unlink()


    print("\n")
    print("=" * 78)
    print(
        f"RUNNING SCENARIO: {scenario_name}"
    )
    print("=" * 78)


    for step, utterance in enumerate(
        history,
        start=1
    ):

        evidence_id = (
            f"{scenario_name}_step_{step:02d}"
        )


        state_before = deepcopy(
            state
        )


        # --------------------------------------------------------
        # 1. Fast semantic interpretation z_t
        # --------------------------------------------------------

        interpretation, usage = (
            interpret_under_state(
                state_before,
                utterance
            )
        )


        # --------------------------------------------------------
        # 2. Explicit longitudinal state update
        # --------------------------------------------------------

        state_after, transition = (
            update_change_tracker(

                state_before,

                interpretation,

                evidence_id,

                step
            )
        )


        # --------------------------------------------------------
        # 3. Explicit heuristic alpha_t
        # --------------------------------------------------------

        alpha_t = (
            compute_revision_openness(

                state_after,

                interpretation
            )
        )


        # --------------------------------------------------------
        # 4. Persist state for next timestep
        # --------------------------------------------------------

        state = state_after


        # --------------------------------------------------------
        # 5. Full auditable record
        # --------------------------------------------------------

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


        record = {

            "scenario":
                scenario_name,

            "step":
                step,

            "evidence_id":
                evidence_id,

            "utterance":
                utterance,

            "state_before":
                state_before,

            "interpretation":
                interpretation.model_dump(),

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

                "tracker_strength":
                    tracker[
                        "confidence"
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
                    )
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
        # Human-readable console trace
        # --------------------------------------------------------

        print(
            f"\n[{step:02d}] "
            f"{utterance}"
        )

        print(
            "     relation      :",
            interpretation.relation_to_established_state
        )

        print(
            "     event          :",
            interpretation.event_status
        )

        print(
            "     change dir     :",
            interpretation.candidate_change_direction
        )

        print(
            "     evidence       :",
            round(
                interpretation.evidence_for_longitudinal_change,
                3
            )
        )

        print(
            "     confidence     :",
            round(
                interpretation.confidence,
                3
            )
        )

        print(
            "     change score   :",
            round(
                tracker[
                    "score"
                ],
                3
            )
        )

        print(
            "     tracker status :",
            tracker[
                "status"
            ]
        )

        print(
            "     alpha_t        :",
            round(
                alpha_t,
                3
            )
        )

        if PRINT_RATIONALE:

            print(
                "     rationale      :",
                interpretation.rationale
            )


    # ------------------------------------------------------------
    # Save final state separately
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
# Run both histories
# ================================================================

final_state_transient, records_transient = (
    run_history(

        scenario_name="transient",

        history=HISTORY_TRANSIENT
    )
)


final_state_sustained, records_sustained = (
    run_history(

        scenario_name="sustained_change",

        history=HISTORY_SUSTAINED_CHANGE
    )
)


# %% ============================================================
# Final-state comparison
# ================================================================

print("\n")
print("=" * 78)
print("FINAL STATE COMPARISON")
print("=" * 78)


def print_final_state_summary(
    label: str,
    state: dict
):

    domain = state[
        STATE_KEY
    ]

    tracker = domain[
        "change_tracker"
    ]

    provisional = domain[
        "provisional_change"
    ]


    print(
        f"\n--- {label} ---"
    )

    print(
        "change score               :",
        round(
            tracker[
                "score"
            ],
            3
        )
    )

    print(
        "tracker confidence          :",
        round(
            tracker[
                "confidence"
            ],
            3
        )
    )

    print(
        "tracker status              :",
        tracker[
            "status"
        ]
    )

    print(
        "qualifying evidence count   :",
        tracker[
            "qualifying_evidence_count"
        ]
    )

    print(
        "provisional active          :",
        provisional is not None
    )

    if provisional is not None:

        print(
            "provisional candidate       :",
            provisional[
                "candidate"
            ]
        )

        print(
            "provisional confidence      :",
            round(
                provisional[
                    "confidence"
                ],
                3
            )
        )


print_final_state_summary(
    "TRANSIENT HISTORY",
    final_state_transient
)

print_final_state_summary(
    "SUSTAINED-CHANGE HISTORY",
    final_state_sustained
)


# %% ============================================================
# SAME final probe under history-formed states
#
#
# This is now the real test:
#
#     H_A -> g_A
#     H_B -> g_B
#
#     SAME x_probe
#
#     g_A + x_probe -> z_A
#     g_B + x_probe -> z_B
#
# No manual state intervention.
# ================================================================

probe_transient, probe_usage_transient = (
    interpret_under_state(

        final_state_transient,

        FINAL_PROBE
    )
)


probe_sustained, probe_usage_sustained = (
    interpret_under_state(

        final_state_sustained,

        FINAL_PROBE
    )
)


print("\n")
print("=" * 78)
print("FINAL SAME-UTTERANCE PROBE")
print("=" * 78)

print(
    "\nSAME PROBE:"
)

print(
    FINAL_PROBE
)


print(
    "\n--- AFTER TRANSIENT HISTORY ---"
)

print(
    probe_transient.model_dump_json(
        indent=2
    )
)


print(
    "\n--- AFTER SUSTAINED-CHANGE HISTORY ---"
)

print(
    probe_sustained.model_dump_json(
        indent=2
    )
)


# %% ============================================================
# Save probe comparison
# ================================================================

probe_record = {

    "probe":
        FINAL_PROBE,

    "after_transient_history":
        probe_transient.model_dump(),

    "after_sustained_change_history":
        probe_sustained.model_dump(),

    "usage": {

        "transient":
            probe_usage_transient,

        "sustained_change":
            probe_usage_sustained
    }
}


probe_path = (

    OUTPUT_DIR
    / "final_probe_comparison.json"
)

with probe_path.open(
    "w",
    encoding="utf-8"
) as f:

    json.dump(

        probe_record,

        f,

        indent=2,

        ensure_ascii=False
    )


# %% ============================================================
# Plot helpers
#
# Data come directly from the JSONL-equivalent records.
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


    elif metric == "tracker_strength":

        return [
            record[
                "summary"
            ][
                "tracker_strength"
            ]
            for record in records
        ]


    elif metric == "alpha_t":

        return [
            record[
                "alpha_t"
            ]
            for record in records
        ]


    else:

        raise ValueError(
            f"Unknown metric: {metric}"
        )


def plot_metric_comparison(
    metric: str,
    ylabel: str,
    filename: str
):

    transient_values = (
        extract_metric(
            records_transient,
            metric
        )
    )

    sustained_values = (
        extract_metric(
            records_sustained,
            metric
        )
    )


    steps = list(
        range(
            1,
            len(
                transient_values
            )
            + 1
        )
    )


    plt.figure(
        figsize=(8, 5)
    )


    plt.plot(
        steps,
        transient_values,
        marker="o",
        label="Transient history"
    )


    plt.plot(
        steps,
        sustained_values,
        marker="o",
        label="Sustained-change history"
    )


    plt.axhline(

        ACTIVE_THRESHOLD,

        linestyle="--",

        label="Activation threshold"

    ) if metric == "change_score" else None


    plt.xlabel(
        "Timestep"
    )

    plt.ylabel(
        ylabel
    )

    plt.title(
        f"{ylabel} across longitudinal history"
    )

    plt.xticks(
        steps
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
# Generate trajectory plots
# ================================================================

plot_metric_comparison(

    metric="change_score",

    ylabel="Accumulated change score",

    filename="change_score_trajectory.png"
)


plot_metric_comparison(

    metric="tracker_strength",

    ylabel="Provisional-change confidence",

    filename="tracker_strength_trajectory.png"
)


plot_metric_comparison(

    metric="alpha_t",

    ylabel="Revision openness alpha_t",

    filename="alpha_trajectory.png"
)


# %% ============================================================
# Final qualitative acceptance check
# ================================================================

transient_domain = (
    final_state_transient[
        STATE_KEY
    ]
)

sustained_domain = (
    final_state_sustained[
        STATE_KEY
    ]
)


transient_active = (

    transient_domain[
        "provisional_change"
    ]
    is not None
)


sustained_active = (

    sustained_domain[
        "provisional_change"
    ]
    is not None
)


print("\n")
print("=" * 78)
print("QUALITATIVE ACCEPTANCE CHECK")
print("=" * 78)


print(
    "\nTransient history provisional active:",
    transient_active
)

print(
    "Sustained history provisional active:",
    sustained_active
)


if (

    not transient_active

    and

    sustained_active

):

    print(
        "\nPASS:"
        "\nHistory alone produced different longitudinal states."
    )

else:

    print(
        "\nNOT YET CLEAN:"
        "\nInspect the JSONL trajectory before changing thresholds."
    )


print(
    "\nOutputs saved to:"
)

print(
    OUTPUT_DIR
)