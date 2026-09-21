# -*- coding: utf-8 -*-
"""
주어진 g가 z를 condition할 수 있나? -> yes.
(아직까진 history state_b를 손으로 만들어주고 있음, 이건 02단계)
"""

# %% ============================================================
# 01_state_conditioning_demo.py
#
# Minimal causal test:
#
#     SAME present evidence x_t
#             +
#     DIFFERENT slow state g_(t-1)
#             ↓
#     DIFFERENT interpretation z_t
#             ↓
#     DIFFERENT revision openness alpha_t
#
# Important architectural separation:
#
#     LLM        -> semantic interpretation z_t
#     Python     -> explicit revision rule alpha_t
#     State      -> NOT directly rewritten by the response model
#
# This is NOT yet a calibrated state updater.
# alpha_t is only an explicit demo heuristic.
# ================================================================

# %% Imports

from pathlib import Path
import os
import json
from typing import Literal, Optional

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field


# %% Environment

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

MODEL = os.environ["OPENAI_MODEL"]

client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"]
)


# %% ============================================================
# Structured interpretation schema
#
# IMPORTANT:
# The LLM does NOT decide alpha_t directly.
#
# It only reports semantic properties of the evidence.
# ================================================================

class ConditionedInterpretation(BaseModel):

    interpretation: str = Field(
        description=(
            "Concise semantic interpretation of the new utterance "
            "in light of the existing longitudinal user state."
        )
    )

    event_status: Literal[
        "transient_event",
        "persistent_event",
        "ambiguous"
    ] = Field(
        description=(
            "Status of the CURRENT observation itself. "
            "This is separate from whether it supports a longer-term change."
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

    relation_to_provisional_change: Literal[
        "supports",
        "contradicts",
        "no_active_change",
        "ambiguous"
    ] = Field(
        description=(
            "Whether the current evidence supports an already-emerging "
            "candidate change in the longitudinal state."
        )
    )

    evidence_for_longitudinal_change: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "How strongly the current evidence contributes to the hypothesis "
            "that the established baseline may genuinely be changing. "
            "This is evidence strength, NOT the final state-update rate."
        )
    )

    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in the interpretation."
    )

    rationale: str = Field(
        description=(
            "Brief explanation distinguishing the current event from "
            "the longer-term longitudinal inference."
        )
    )


# %% ============================================================
# Slow states
#
# CRITICAL DESIGN:
#
# The ESTABLISHED baseline is IDENTICAL across A and B.
#
# The only intervention is whether accumulated recent history
# has already created a provisional-change hypothesis.
#
# This makes the causal comparison cleaner.
# ================================================================

state_a = {
    "morning_walk": {

        "established_state": {
            "value": (
                "Usually takes and enjoys a morning walk "
                "around 8 AM."
            ),
            "confidence": 0.95,
            "stability": 0.95,
            "last_confirmed": "yesterday"
        },

        "provisional_change": None
    }
}


state_b = {
    "morning_walk": {

        "established_state": {
            "value": (
                "Usually takes and enjoys a morning walk "
                "around 8 AM."
            ),
            "confidence": 0.95,
            "stability": 0.95,
            "last_confirmed": "10 days ago"
        },

        "provisional_change": {
            "active": True,
            "candidate": (
                "Reduced interest in morning walks "
                "and in going outside."
            ),
            "duration_days": 10,
            "evidence_count": 8,
            "confidence": 0.72
        }
    }
}


# %% Same present evidence in both conditions

current_utterance = "I don't feel like going outside today."


# %% ============================================================
# LLM interpretation function
#
# Conceptually:
#
#     z_t = f(x_t, g_(t-1))
#
# No persistent state is modified here.
# ================================================================

def interpret_under_state(
    current_state: dict,
    utterance: str
) -> ConditionedInterpretation:

    system_prompt = """
You are the FAST INTERPRETATION component of a longitudinal
user-modeling architecture.

You receive:

1. an ESTABLISHED USER STATE:
   a relatively stable, historically supported model of the person,

2. an optional PROVISIONAL CHANGE:
   an emerging hypothesis that the established state may currently
   be changing,

3. a NEW UTTERANCE:
   present conversational evidence.

Your task is to INTERPRET the new evidence.

You must NOT rewrite the persistent user state.
You must NOT choose the final state-update rate.

Keep two levels of inference explicitly separate:

A. EVENT LEVEL
   What does this particular utterance indicate right now?
   A statement about "today" may still be a transient event.

B. LONGITUDINAL LEVEL
   Does this event support or contradict the established baseline?
   Does it also reinforce an already-emerging change hypothesis?

Important principles:

- A single contradictory observation should not automatically
  overwrite a strongly supported established state.

- A current event may itself be transient while simultaneously
  reinforcing a longer-running pattern of change.

- If there is no active provisional-change hypothesis, do not invent
  one merely because of one ambiguous or contradictory observation.

- If accumulated history already contains an active provisional
  change and the new evidence is consistent with it, report that
  relationship explicitly.

Return only the requested structured interpretation.
"""

    user_prompt = f"""
CURRENT SLOW USER STATE:

{json.dumps(current_state, indent=2)}

NEW CURRENT EVIDENCE:

{utterance}
"""

    response = client.responses.parse(
        model=MODEL,
        reasoning={"effort": "minimal"},
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

    return response.output_parsed


# %% ============================================================
# Explicit state-dynamics functions
#
# LLM provides semantic interpretation.
# Python implements revision dynamics.
#
# These weights are NOT scientifically calibrated.
# They are deliberately transparent demo heuristics.
# ================================================================

def get_provisional_trend_strength(
    current_state: dict,
    state_key: str = "morning_walk"
) -> float:

    provisional = current_state[state_key].get(
        "provisional_change"
    )

    if provisional is None:
        return 0.0

    if not provisional.get("active", False):
        return 0.0

    confidence = provisional.get(
        "confidence",
        0.0
    )

    duration_score = min(
        provisional.get("duration_days", 0) / 14.0,
        1.0
    )

    evidence_score = min(
        provisional.get("evidence_count", 0) / 10.0,
        1.0
    )

    trend_strength = (
        0.40 * confidence
        + 0.30 * duration_score
        + 0.30 * evidence_score
    )

    return max(
        0.0,
        min(1.0, trend_strength)
    )


def compute_revision_openness(
    current_state: dict,
    interpretation: ConditionedInterpretation,
    state_key: str = "morning_walk"
) -> dict:
    """
    Compute alpha_t explicitly.

    alpha_t represents openness to revising the ESTABLISHED state.

    IMPORTANT:
    This is currently only a transparent heuristic used to test
    the architecture.

    Later alternatives:
        - Bayesian update
        - learned update policy
        - meta-learned plasticity
        - calibrated probabilistic model
    """

    established = current_state[state_key][
        "established_state"
    ]

    established_stability = established.get(
        "stability",
        0.5
    )

    trend_strength = get_provisional_trend_strength(
        current_state,
        state_key
    )

    # Does the new evidence oppose the established baseline?
    baseline_contradiction = (
        1.0
        if interpretation.relation_to_established_state
        == "contradicts"
        else 0.0
    )

    # Does the evidence reinforce an already-emerging change?
    provisional_support = (
        1.0
        if interpretation.relation_to_provisional_change
        == "supports"
        else 0.0
    )

    # Raw revision pressure from interpretable components
    raw_pressure = (
        0.30
        * interpretation.evidence_for_longitudinal_change

        + 0.20
        * interpretation.confidence

        + 0.30
        * trend_strength

        + 0.20
        * provisional_support
    )

    # Evidence that does not contradict the established baseline
    # should exert substantially less pressure to revise it.
    contradiction_gate = (
        0.25
        + 0.75 * baseline_contradiction
    )

    raw_pressure *= contradiction_gate

    # Stable established beliefs resist revision.
    #
    # High stability does NOT make revision impossible;
    # it simply lowers instantaneous openness.
    resistance = (
        0.65 * established_stability
    )

    alpha_t = (
        raw_pressure
        * (1.0 - resistance)
    )

    alpha_t = max(
        0.0,
        min(1.0, alpha_t)
    )

    return {
        "alpha_t": alpha_t,

        "raw_pressure": raw_pressure,

        "established_stability":
            established_stability,

        "trend_strength":
            trend_strength,

        "baseline_contradiction":
            baseline_contradiction,

        "provisional_support":
            provisional_support
    }


# %% ============================================================
# Run State A
# ================================================================

result_a = interpret_under_state(
    state_a,
    current_utterance
)

dynamics_a = compute_revision_openness(
    state_a,
    result_a
)

print("\n")
print("=" * 70)
print("STATE A")
print("=" * 70)

print(
    result_a.model_dump_json(
        indent=2
    )
)

print("\nExplicit dynamics:")

print(
    json.dumps(
        dynamics_a,
        indent=2
    )
)


# %% ============================================================
# Run State B
# ================================================================

result_b = interpret_under_state(
    state_b,
    current_utterance
)

dynamics_b = compute_revision_openness(
    state_b,
    result_b
)

print("\n")
print("=" * 70)
print("STATE B")
print("=" * 70)

print(
    result_b.model_dump_json(
        indent=2
    )
)

print("\nExplicit dynamics:")

print(
    json.dumps(
        dynamics_b,
        indent=2
    )
)


# %% ============================================================
# Minimal causal comparison
# ================================================================

print("\n")
print("=" * 70)
print("CAUSAL COMPARISON")
print("=" * 70)

print("\nSAME CURRENT EVIDENCE:")
print(current_utterance)


print("\n--- STATE A ---")

print(
    "event status                  :",
    result_a.event_status
)

print(
    "relation to established       :",
    result_a.relation_to_established_state
)

print(
    "relation to provisional       :",
    result_a.relation_to_provisional_change
)

print(
    "evidence for change           :",
    round(
        result_a.evidence_for_longitudinal_change,
        3
    )
)

print(
    "interpretation confidence     :",
    round(
        result_a.confidence,
        3
    )
)

print(
    "trend strength                :",
    round(
        dynamics_a["trend_strength"],
        3
    )
)

print(
    "alpha_t                       :",
    round(
        dynamics_a["alpha_t"],
        3
    )
)


print("\n--- STATE B ---")

print(
    "event status                  :",
    result_b.event_status
)

print(
    "relation to established       :",
    result_b.relation_to_established_state
)

print(
    "relation to provisional       :",
    result_b.relation_to_provisional_change
)

print(
    "evidence for change           :",
    round(
        result_b.evidence_for_longitudinal_change,
        3
    )
)

print(
    "interpretation confidence     :",
    round(
        result_b.confidence,
        3
    )
)

print(
    "trend strength                :",
    round(
        dynamics_b["trend_strength"],
        3
    )
)

print(
    "alpha_t                       :",
    round(
        dynamics_b["alpha_t"],
        3
    )
)


# %% Direct contrasts

delta_evidence = (
    result_b.evidence_for_longitudinal_change
    - result_a.evidence_for_longitudinal_change
)

delta_alpha = (
    dynamics_b["alpha_t"]
    - dynamics_a["alpha_t"]
)


print("\n--- DIFFERENCE B - A ---")

print(
    "Δ evidence for change         :",
    round(
        delta_evidence,
        3
    )
)

print(
    "Δ alpha_t                     :",
    round(
        delta_alpha,
        3
    )
)


# %% ============================================================
# Interpretation summary
# ================================================================

print("\n")
print("=" * 70)
print("EXPECTED ARCHITECTURAL SIGNATURE")
print("=" * 70)

print(
    """
The present utterance is identical across conditions.

State A:
    no active longitudinal change hypothesis

State B:
    accumulated history already supports an emerging change

Desired qualitative result:

    A:
        current event may contradict the baseline,
        but provides limited pressure for baseline revision

    B:
        current event may still be transient at the event level,
        while simultaneously reinforcing an emerging longitudinal
        change and producing greater revision openness

Therefore:

    SAME x_t
        +
    DIFFERENT g_(t-1)
        ->
    DIFFERENT z_t
        ->
    DIFFERENT alpha_t

This is the minimal perspective-conditioning mechanism
being tested here.
"""
)