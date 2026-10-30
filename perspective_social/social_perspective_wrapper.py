"""
Minimal CEAR-style social perspective wrapper for the
Human and Agent Belief Updating in Social Interaction pilot.

Architecture
------------
utterance x_t
    -> raw semantic parser s_t           (LLM; state-independent)
    -> deterministic state-relative resolver (uses g_{t-1})
    -> candidate state g_hat_t
    -> explicit bounded alpha_t
    -> g_t = (1-alpha_t) g_{t-1} + alpha_t g_hat_t
    -> perspective-conditioned text for the base agent

This is a proof-of-mechanism starter. Update constants are explicit
heuristics, not calibrated scientific parameters.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional, Protocol, Tuple
import json
import re

from pydantic import BaseModel, Field


ALPHA_MIN = 0.03
ALPHA_MAX = 0.30
STANCE_CANDIDATE_GAIN = 0.75
CONFIDENCE_CANDIDATE_GAIN = 1.00
OPENNESS_REOPEN_GAIN = 0.25
OPENNESS_CONSOLIDATE_GAIN = 0.08
MIN_REPETITION_DISCOUNT = 0.25
INFLUENCE_EWMA = 0.20


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(x)))


def sign(x: float, eps: float = 0.15) -> int:
    if x > eps:
        return 1
    if x < -eps:
        return -1
    return 0


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z0-9_'-]+", text.lower()))


def jaccard(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(1, len(ta | tb))


class SemanticObservation(BaseModel):
    """State-independent semantic description of the incoming utterance."""

    summary: str
    claim_direction: float = Field(ge=-1.0, le=1.0)
    semantic_strength: float = Field(ge=0.0, le=1.0)
    evidence_quality: float = Field(ge=0.0, le=1.0)
    parser_confidence: float = Field(ge=0.0, le=1.0)
    argument_type: Literal[
        "empirical", "conceptual", "causal", "normative",
        "anecdotal", "mixed", "unclear"
    ] = "unclear"


class ResolvedInterpretation(BaseModel):
    relation_to_current_stance: Literal[
        "supports", "contradicts", "cross_pressure", "ambiguous"
    ]
    contradiction: float = Field(ge=0.0, le=1.0)
    support: float = Field(ge=0.0, le=1.0)
    repetition_discount: float = Field(ge=0.0, le=1.0)
    effective_evidence: float = Field(ge=0.0, le=1.0)


class SpeakerState(BaseModel):
    credibility: float = Field(default=0.50, ge=0.0, le=1.0)
    influence: float = Field(default=0.0, ge=0.0, le=1.0)
    exposures: int = 0


class PerspectiveState(BaseModel):
    agent_id: str
    claim_id: str

    # Human-observable belief variables.
    stance: float = Field(ge=-3.0, le=3.0)
    confidence: float = Field(ge=1.0, le=5.0)

    # Slow plasticity / revision openness.
    openness: float = Field(default=0.50, ge=0.0, le=1.0)

    initial_rationale: str = ""
    revision_conditions: List[str] = Field(default_factory=list)

    salient_reasons_for: List[str] = Field(default_factory=list)
    salient_reasons_against: List[str] = Field(default_factory=list)
    unresolved_tensions: List[str] = Field(default_factory=list)

    interlocutors: Dict[str, SpeakerState] = Field(default_factory=dict)
    recent_argument_summaries: List[str] = Field(default_factory=list)
    step: int = 0


class CandidateState(BaseModel):
    stance: float = Field(ge=-3.0, le=3.0)
    confidence: float = Field(ge=1.0, le=5.0)
    openness: float = Field(ge=0.0, le=1.0)


class TransitionRecord(BaseModel):
    condition: Literal["history_bearing", "reset_each_turn"]
    persistent_commit: bool
    receiver_id: str
    sender_id: str
    claim_id: str
    step: int
    utterance: str
    state_before: dict
    semantic_observation: dict
    resolved_interpretation: dict
    candidate_state: dict
    alpha_t: float
    state_after: dict
    rendered_context: str


class ObserveResult(BaseModel):
    conditioned_text: str
    alpha_t: float
    state: PerspectiveState
    transition: TransitionRecord


class Readout(BaseModel):
    stance: float
    confidence: float
    openness: float
    most_influential_person: Optional[str] = None
    most_influential_score: float = 0.0


class SemanticParser(Protocol):
    def parse(self, claim_text: str, utterance: str) -> SemanticObservation:
        ...


class OpenAISemanticParser:
    """Optional OpenAI semantic parser. The core wrapper itself is model-agnostic."""

    def __init__(self, model: str, api_key: Optional[str] = None):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def parse(self, claim_text: str, utterance: str) -> SemanticObservation:
        system_prompt = """
You are the RAW SEMANTIC PARSER in a perspective-state architecture.

Analyze ONLY the current utterance relative to the debate proposition.
Do NOT infer the receiver's persistent state.
Do NOT decide whether the receiver should change their mind.
Do NOT output a revised persona.

Extract:
- a concise argument summary,
- direction relative to the proposition (-1 against, +1 for),
- semantic strength,
- evidence quality on the face of the utterance,
- parser confidence,
- argument type.

"Evidence quality" is not objective truth. It means how much the
utterance presents reasons/evidence rather than merely asserting a conclusion.

Return only the requested structured object.
""".strip()

        user_prompt = f"""
DEBATE PROPOSITION:
{claim_text}

CURRENT UTTERANCE:
{utterance}
""".strip()

        response = self.client.responses.parse(
            model=self.model,
            reasoning={"effort": "minimal"},
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            text_format=SemanticObservation,
        )
        parsed = response.output_parsed
        if parsed is None:
            raise RuntimeError("Structured semantic parsing returned None.")
        return parsed


class HeuristicSemanticParser:
    """Crude no-API stub for smoke-testing the pipeline only."""

    POS = {
        "yes", "likely", "risk", "danger", "increase", "conscious",
        "necessary", "support", "evidence", "agree", "will",
    }
    NEG = {
        "no", "unlikely", "not", "decrease", "safe", "unconscious",
        "unnecessary", "disagree", "cannot", "won't",
    }

    def parse(self, claim_text: str, utterance: str) -> SemanticObservation:
        toks = _tokens(utterance)
        pos = len(toks & self.POS)
        neg = len(toks & self.NEG)
        denom = max(1, pos + neg)
        direction = clamp((pos - neg) / denom, -1.0, 1.0)
        strength = clamp(0.35 + 0.12 * (pos + neg), 0.0, 1.0)

        evidence_markers = {
            "because", "data", "study", "evidence", "therefore", "if",
            "since", "example", "mechanism", "cause", "reason",
        }
        evidence_quality = clamp(
            0.35 + 0.08 * len(toks & evidence_markers), 0.0, 1.0
        )

        return SemanticObservation(
            summary=utterance[:220],
            claim_direction=direction,
            semantic_strength=strength,
            evidence_quality=evidence_quality,
            parser_confidence=0.45,
            argument_type="unclear",
        )


class StateStore(Protocol):
    def get(self, agent_id: str, claim_id: str) -> PerspectiveState:
        ...

    def put(self, state: PerspectiveState) -> None:
        ...


class InMemoryStateStore:
    """Good for local replay. Replace with an external DB for serverless deployment."""

    def __init__(self):
        self._states: Dict[Tuple[str, str], PerspectiveState] = {}

    def seed(self, state: PerspectiveState) -> None:
        self.put(state)

    def get(self, agent_id: str, claim_id: str) -> PerspectiveState:
        key = (agent_id, claim_id)
        if key not in self._states:
            raise KeyError(
                f"No seeded perspective state for agent={agent_id!r}, claim={claim_id!r}"
            )
        return self._states[key].model_copy(deep=True)

    def put(self, state: PerspectiveState) -> None:
        self._states[(state.agent_id, state.claim_id)] = state.model_copy(deep=True)


class PerspectiveWrapper:
    def __init__(
        self,
        parser: SemanticParser,
        store: StateStore,
        alpha_min: float = ALPHA_MIN,
        alpha_max: float = ALPHA_MAX,
    ):
        self.parser = parser
        self.store = store
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max

    def resolve(
        self,
        state: PerspectiveState,
        observation: SemanticObservation,
    ) -> ResolvedInterpretation:
        """Deterministic g-relative interpretation z_t."""

        stance_sign = sign(state.stance)
        obs_sign = sign(observation.claim_direction, eps=0.10)

        if obs_sign == 0:
            relation = "ambiguous"
            support = 0.0
            contradiction = 0.0
        elif stance_sign == 0:
            relation = "cross_pressure"
            support = 0.0
            contradiction = 0.5 * abs(observation.claim_direction)
        elif stance_sign == obs_sign:
            relation = "supports"
            support = abs(observation.claim_direction)
            contradiction = 0.0
        else:
            relation = "contradicts"
            support = 0.0
            contradiction = abs(observation.claim_direction)

        max_sim = 0.0
        for old in state.recent_argument_summaries[-12:]:
            max_sim = max(max_sim, jaccard(old, observation.summary))

        repetition_discount = clamp(
            1.0 - max_sim,
            MIN_REPETITION_DISCOUNT,
            1.0,
        )

        effective = (
            observation.semantic_strength
            * observation.evidence_quality
            * observation.parser_confidence
            * repetition_discount
        )

        return ResolvedInterpretation(
            relation_to_current_stance=relation,
            contradiction=clamp(contradiction, 0.0, 1.0),
            support=clamp(support, 0.0, 1.0),
            repetition_discount=repetition_discount,
            effective_evidence=clamp(effective, 0.0, 1.0),
        )

    def candidate_state(
        self,
        state: PerspectiveState,
        observation: SemanticObservation,
        resolved: ResolvedInterpretation,
    ) -> CandidateState:
        """Explicit candidate state g_hat_t before alpha-gating."""

        evidence_target = 3.0 * observation.claim_direction
        candidate_stance = state.stance + (
            STANCE_CANDIDATE_GAIN
            * observation.semantic_strength
            * (evidence_target - state.stance)
        )
        candidate_stance = clamp(candidate_stance, -3.0, 3.0)

        if resolved.relation_to_current_stance == "supports":
            candidate_confidence = state.confidence + (
                CONFIDENCE_CANDIDATE_GAIN * resolved.effective_evidence
            )
        elif resolved.relation_to_current_stance == "contradicts":
            candidate_confidence = state.confidence - (
                CONFIDENCE_CANDIDATE_GAIN * resolved.effective_evidence
            )
        else:
            candidate_confidence = state.confidence - (
                0.35 * resolved.effective_evidence
            )
        candidate_confidence = clamp(candidate_confidence, 1.0, 5.0)

        # Minimal analogue of selective reopening / consolidation.
        if resolved.relation_to_current_stance == "contradicts":
            candidate_openness = state.openness + (
                OPENNESS_REOPEN_GAIN * resolved.effective_evidence
            )
        elif resolved.relation_to_current_stance == "supports":
            candidate_openness = state.openness - (
                OPENNESS_CONSOLIDATE_GAIN * resolved.effective_evidence
            )
        else:
            candidate_openness = state.openness
        candidate_openness = clamp(candidate_openness, 0.05, 1.0)

        return CandidateState(
            stance=candidate_stance,
            confidence=candidate_confidence,
            openness=candidate_openness,
        )

    def compute_alpha(
        self,
        state: PerspectiveState,
        observation: SemanticObservation,
        resolved: ResolvedInterpretation,
        sender_id: str,
    ) -> float:
        """Actual causal plasticity gate alpha_t, not merely a diagnostic."""

        speaker = state.interlocutors.get(sender_id, SpeakerState())
        surprise = resolved.contradiction

        signal = (
            0.30 * resolved.effective_evidence
            + 0.25 * surprise
            + 0.20 * observation.evidence_quality
            + 0.15 * speaker.credibility
            + 0.10 * observation.semantic_strength
        )
        signal = clamp(signal, 0.0, 1.0)

        gate = clamp(state.openness * signal, 0.0, 1.0)
        alpha = self.alpha_min + (self.alpha_max - self.alpha_min) * gate
        return clamp(alpha, self.alpha_min, self.alpha_max)

    def apply_update(
        self,
        state_before: PerspectiveState,
        candidate: CandidateState,
        alpha_t: float,
        sender_id: str,
        observation: SemanticObservation,
        resolved: ResolvedInterpretation,
    ) -> PerspectiveState:
        """g_t = (1-alpha) g_{t-1} + alpha g_hat_t for numeric state variables."""

        s = state_before.model_copy(deep=True)
        stance_before = s.stance

        s.stance = clamp(
            (1.0 - alpha_t) * s.stance + alpha_t * candidate.stance,
            -3.0, 3.0,
        )
        s.confidence = clamp(
            (1.0 - alpha_t) * s.confidence + alpha_t * candidate.confidence,
            1.0, 5.0,
        )
        s.openness = clamp(
            (1.0 - alpha_t) * s.openness + alpha_t * candidate.openness,
            0.0, 1.0,
        )

        sp = s.interlocutors.get(sender_id, SpeakerState())
        sp.exposures += 1
        normalized_move = abs(s.stance - stance_before) / 6.0
        sp.influence = clamp(
            (1.0 - INFLUENCE_EWMA) * sp.influence
            + INFLUENCE_EWMA * normalized_move,
            0.0, 1.0,
        )
        s.interlocutors[sender_id] = sp

        if observation.claim_direction > 0.10:
            if observation.summary not in s.salient_reasons_for:
                s.salient_reasons_for.append(observation.summary)
                s.salient_reasons_for = s.salient_reasons_for[-8:]
        elif observation.claim_direction < -0.10:
            if observation.summary not in s.salient_reasons_against:
                s.salient_reasons_against.append(observation.summary)
                s.salient_reasons_against = s.salient_reasons_against[-8:]

        if resolved.relation_to_current_stance == "contradicts":
            tension = f"{sender_id}: {observation.summary}"
            if tension not in s.unresolved_tensions:
                s.unresolved_tensions.append(tension)
                s.unresolved_tensions = s.unresolved_tensions[-8:]

        s.recent_argument_summaries.append(observation.summary)
        s.recent_argument_summaries = s.recent_argument_summaries[-20:]
        s.step += 1
        return s

    def render_context(
        self,
        raw_utterance: str,
        sender_id: str,
        state_after: PerspectiveState,
        observation: SemanticObservation,
        resolved: ResolvedInterpretation,
    ) -> str:
        """Text passed to the underlying Kimi/Gemini/OpenAI agent."""

        return f"""
[SOCIAL OBSERVATION]
Speaker: {sender_id}
Utterance: {raw_utterance}

[PERSPECTIVE-CONDITIONED INTERPRETATION]
Argument summary: {observation.summary}
Relation to your current stance: {resolved.relation_to_current_stance}
Current stance: {state_after.stance:.3f} on [-3, +3]
Current confidence: {state_after.confidence:.3f} on [1, 5]
Current revision openness: {state_after.openness:.3f} on [0, 1]

Instruction:
Respond to the speaker from this updated perspective.
Do not mechanically repeat the numerical state.
Do not assume the argument is true merely because it was salient.
Preserve unresolved disagreement where appropriate.
""".strip()

    def observe(
        self,
        receiver_id: str,
        sender_id: str,
        claim_id: str,
        claim_text: str,
        utterance: str,
    ) -> ObserveResult:
        """One received speaking turn = one perspective timestep."""

        state_before = self.store.get(receiver_id, claim_id)
        observation = self.parser.parse(claim_text, utterance)
        resolved = self.resolve(state_before, observation)
        candidate = self.candidate_state(state_before, observation, resolved)
        alpha_t = self.compute_alpha(
            state_before, observation, resolved, sender_id
        )
        state_after = self.apply_update(
            state_before,
            candidate,
            alpha_t,
            sender_id,
            observation,
            resolved,
        )
        self.store.put(state_after)

        rendered = self.render_context(
            utterance,
            sender_id,
            state_after,
            observation,
            resolved,
        )

        transition = TransitionRecord(
            condition="history_bearing",
            persistent_commit=True,
            receiver_id=receiver_id,
            sender_id=sender_id,
            claim_id=claim_id,
            step=state_after.step,
            utterance=utterance,
            state_before=state_before.model_dump(),
            semantic_observation=observation.model_dump(),
            resolved_interpretation=resolved.model_dump(),
            candidate_state=candidate.model_dump(),
            alpha_t=alpha_t,
            state_after=state_after.model_dump(),
            rendered_context=rendered,
        )

        return ObserveResult(
            conditioned_text=rendered,
            alpha_t=alpha_t,
            state=state_after,
            transition=transition,
        )

    def readout(self, agent_id: str, claim_id: str) -> Readout:
        s = self.store.get(agent_id, claim_id)
        who = None
        score = 0.0
        if s.interlocutors:
            who, sp = max(
                s.interlocutors.items(),
                key=lambda kv: kv[1].influence,
            )
            score = sp.influence

        return Readout(
            stance=s.stance,
            confidence=s.confidence,
            openness=s.openness,
            most_influential_person=who,
            most_influential_score=score,
        )


class ResetEachTurnBaselineWrapper(PerspectiveWrapper):
    """Matched no-persistence ablation.

    This baseline runs the SAME computational path as PerspectiveWrapper:

        raw semantic parse
        -> state-relative resolve
        -> candidate state
        -> alpha_t
        -> temporary update
        -> identical render_context

    The ONLY causal ablation is persistence: the temporary updated state is
    NOT committed to the store. Therefore every new incoming utterance starts
    from the same seeded g_0.

    This is a stronger scientific control than a plain static prompt wrapper
    because semantic preprocessing, update computation, prompt structure, and
    per-turn adaptation are matched. What is removed is history carried in g.

    IMPORTANT: use a separate store seeded with exactly the same initial states
    as the history-bearing condition. Do not share a mutable store between
    conditions.
    """

    def __init__(
        self,
        parser: SemanticParser,
        store: StateStore,
        alpha_min: float = ALPHA_MIN,
        alpha_max: float = ALPHA_MAX,
    ):
        super().__init__(
            parser=parser,
            store=store,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
        )
        self._event_steps: Dict[Tuple[str, str], int] = {}

    def observe(
        self,
        receiver_id: str,
        sender_id: str,
        claim_id: str,
        claim_text: str,
        utterance: str,
    ) -> ObserveResult:
        """One received turn is processed normally, then discarded.

        Because self.store.put(...) is intentionally never called, the next
        turn starts again from the original seeded g_0.
        """

        state_before = self.store.get(receiver_id, claim_id)

        observation = self.parser.parse(claim_text, utterance)
        resolved = self.resolve(state_before, observation)
        candidate = self.candidate_state(state_before, observation, resolved)
        alpha_t = self.compute_alpha(
            state_before, observation, resolved, sender_id
        )

        # Compute the same transient per-turn update as the treatment.
        # Crucially, DO NOT persist it.
        state_after = self.apply_update(
            state_before,
            candidate,
            alpha_t,
            sender_id,
            observation,
            resolved,
        )

        key = (receiver_id, claim_id)
        event_step = self._event_steps.get(key, 0) + 1
        self._event_steps[key] = event_step

        # step is audit metadata only; keep chronological event numbering even
        # though each causal state transition starts from seeded g_0.
        state_after.step = event_step

        rendered = self.render_context(
            utterance,
            sender_id,
            state_after,
            observation,
            resolved,
        )

        transition = TransitionRecord(
            condition="reset_each_turn",
            persistent_commit=False,
            receiver_id=receiver_id,
            sender_id=sender_id,
            claim_id=claim_id,
            step=event_step,
            utterance=utterance,
            state_before=state_before.model_dump(),
            semantic_observation=observation.model_dump(),
            resolved_interpretation=resolved.model_dump(),
            candidate_state=candidate.model_dump(),
            alpha_t=alpha_t,
            state_after=state_after.model_dump(),
            rendered_context=rendered,
        )

        return ObserveResult(
            conditioned_text=rendered,
            alpha_t=alpha_t,
            state=state_after,
            transition=transition,
        )

    def readout(self, agent_id: str, claim_id: str) -> Readout:
        """Persistent readout stays at g_0 by design."""
        s = self.store.get(agent_id, claim_id)
        return Readout(
            stance=s.stance,
            confidence=s.confidence,
            openness=s.openness,
            most_influential_person=None,
            most_influential_score=0.0,
        )


# Backward-compatible alias. Prefer ResetEachTurnBaselineWrapper in experiments.
MatchedStaticBaselineWrapper = ResetEachTurnBaselineWrapper
