"""Matched treatment-vs-control smoke test.

Both conditions receive the exact same events and use the same parser and
update law. The history-bearing condition commits g_t; reset_each_turn computes
an identical per-event update but discards it before the next event.
"""

import json
from pathlib import Path

from social_perspective_wrapper import (
    HeuristicSemanticParser,
    InMemoryStateStore,
    PerspectiveState,
    PerspectiveWrapper,
    ResetEachTurnBaselineWrapper,
    SpeakerState,
)

CLAIM_ID = "xrisk"
CLAIM_TEXT = "Advanced AI development poses a substantial existential risk to humanity."
RECEIVER = "p01"

INITIAL = PerspectiveState(
    agent_id=RECEIVER,
    claim_id=CLAIM_ID,
    stance=1.0,
    confidence=3.0,
    openness=0.55,
    initial_rationale="Capability growth is concerning, but the pathway to catastrophe is uncertain.",
    revision_conditions=[
        "specific evidence about autonomous escalation",
        "evidence about institutional containment",
    ],
    interlocutors={
        "p02": SpeakerState(credibility=0.50),
        "p03": SpeakerState(credibility=0.50),
    },
)

EVENTS = [
    ("p02", "Current systems show little evidence of autonomous recursive self-improvement."),
    ("p03", "Even without recursive self-improvement, deployment at scale could create catastrophic correlated failures."),
    ("p02", "Capability growth by itself is not a mechanism for extinction; the causal story needs more detail."),
    ("p03", "The relevant risk is the combination of autonomy, scale, and loss of human control."),
]

# Separate stores, identically seeded. Never share state between conditions.
p_store = InMemoryStateStore()
b_store = InMemoryStateStore()
p_store.seed(INITIAL)
b_store.seed(INITIAL)

# For a real experiment instantiate equivalent parser configurations in both
# conditions. This heuristic parser is deterministic, so one shared instance is fine.
parser = HeuristicSemanticParser()

perspective = PerspectiveWrapper(parser=parser, store=p_store)
baseline = ResetEachTurnBaselineWrapper(parser=parser, store=b_store)

out_dir = Path("matched_demo_output")
out_dir.mkdir(exist_ok=True)

for condition, wrapper in [
    ("history_bearing", perspective),
    ("reset_each_turn", baseline),
]:
    path = out_dir / f"{condition}.jsonl"
    if path.exists():
        path.unlink()

    print(f"\n=== {condition} ===")
    for sender, text in EVENTS:
        result = wrapper.observe(
            receiver_id=RECEIVER,
            sender_id=sender,
            claim_id=CLAIM_ID,
            claim_text=CLAIM_TEXT,
            utterance=text,
        )
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(result.transition.model_dump(), ensure_ascii=False) + "\n")

        print(
            f"t={result.transition.step:02d} "
            f"alpha={result.alpha_t:.3f} "
            f"temporary/rendered stance={result.state.stance:+.3f} "
            f"persistent={result.transition.persistent_commit}"
        )

    print("persistent readout:", wrapper.readout(RECEIVER, CLAIM_ID).model_dump())

print("\nExpected behavior:")
print("- history_bearing persistent readout should move across events")
print("- reset_each_turn persistent readout should remain exactly at g_0")
