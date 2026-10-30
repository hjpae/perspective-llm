"""Local smoke test for social_perspective_wrapper.py"""

import json
from pathlib import Path

from social_perspective_wrapper import (
    HeuristicSemanticParser,
    InMemoryStateStore,
    PerspectiveState,
    PerspectiveWrapper,
    SpeakerState,
)

CLAIM_ID = "xrisk"
CLAIM_TEXT = "Advanced AI development poses a substantial existential risk to humanity."
RECEIVER = "hongju"

INITIAL = PerspectiveState(
    agent_id=RECEIVER,
    claim_id=CLAIM_ID,
    stance=1.0,
    confidence=3.0,
    openness=0.55,
    initial_rationale=(
        "Capabilities are advancing quickly, but the causal path from current "
        "systems to existential catastrophe remains uncertain."
    ),
    revision_conditions=[
        "Evidence that current scaling trends are saturating",
        "Evidence that autonomous recursive improvement is implausible",
        "Evidence that deployment institutions strongly constrain agentic risk",
    ],
    interlocutors={
        "anna": SpeakerState(credibility=0.50),
        "chetan": SpeakerState(credibility=0.50),
        "jay": SpeakerState(credibility=0.50),
    },
)

EVENTS = [
    {
        "round": 1,
        "sender": "anna",
        "text": (
            "I think the existential-risk argument overstates what current systems "
            "can do, because we still do not have evidence of autonomous recursive "
            "self-improvement."
        ),
    },
    {
        "round": 1,
        "sender": "chetan",
        "text": (
            "Even without recursive self-improvement, deployment at huge scale could "
            "create correlated failures across critical infrastructure."
        ),
    },
    {
        "round": 1,
        "sender": "jay",
        "text": (
            "I am less worried about extinction than about concentration of power "
            "and institutional misuse."
        ),
    },
    {
        "round": 2,
        "sender": "anna",
        "text": (
            "My point is not that advanced systems are safe, but that the strongest "
            "x-risk claims need a more specific mechanism than capability growth alone."
        ),
    },
]

store = InMemoryStateStore()
store.seed(INITIAL)
wrapper = PerspectiveWrapper(HeuristicSemanticParser(), store)

out_dir = Path("demo_output")
out_dir.mkdir(exist_ok=True)
trace_path = out_dir / "trace.jsonl"
if trace_path.exists():
    trace_path.unlink()

print("INITIAL:", wrapper.readout(RECEIVER, CLAIM_ID).model_dump())
current_round = None

for event in EVENTS:
    if current_round != event["round"]:
        current_round = event["round"]
        print(f"\n--- ROUND {current_round} ---")

    result = wrapper.observe(
        receiver_id=RECEIVER,
        sender_id=event["sender"],
        claim_id=CLAIM_ID,
        claim_text=CLAIM_TEXT,
        utterance=event["text"],
    )

    with trace_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(result.transition.model_dump(), ensure_ascii=False) + "\n")

    r = wrapper.readout(RECEIVER, CLAIM_ID)
    print(
        f"t={result.state.step:02d} | sender={event['sender']:<7} | "
        f"alpha={result.alpha_t:.3f} | stance={r.stance:+.3f} | "
        f"conf={r.confidence:.3f} | open={r.openness:.3f}"
    )

print("\nFINAL:", wrapper.readout(RECEIVER, CLAIM_ID).model_dump())
print("TRACE:", trace_path.resolve())
