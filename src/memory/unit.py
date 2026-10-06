"""
Condition-action-effect memory units for Seam.

This module is intentionally environment-neutral. Environment adapters provide
normalization and type mappings; memory units store reusable action-effect
rules with evidence, belief state, usage, and provenance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from hashlib import sha1
from typing import Any


class MemoryLevel(Enum):
    HYPOTHESIS = "M0"
    STABLE = "M1"
    ABSTRACT = "M2"


class BeliefState(Enum):
    HYPOTHESIS = "hypothesis"
    STABLE = "stable"
    REVISING = "revising"


class VerificationResult(Enum):
    SUPPORT = "support"
    CONTRADICT = "contradict"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ActionSchema:
    action_type: str
    object_type: str = ""
    receptacle_type: str = ""
    tool_type: str = ""
    raw: str = ""

    def key(self) -> str:
        return "|".join((
            self.action_type,
            self.object_type,
            self.receptacle_type,
            self.tool_type,
        ))


@dataclass(frozen=True)
class Effect:
    effect_type: str
    object_type: str = ""
    receptacle_type: str = ""
    value: str = ""
    text: str = ""

    def key(self) -> str:
        return "|".join((
            self.effect_type,
            self.object_type,
            self.receptacle_type,
            self.value,
        ))


@dataclass(frozen=True)
class Condition:
    task_type: str = ""
    target_obj_type: str = ""
    current_recep_type: str = ""
    holding: bool | None = None
    processed: bool | None = None
    processing: str = ""
    object_category: str = ""
    receptacle_category: str = ""
    predicates: tuple[str, ...] = ()

    def key(self) -> str:
        return "|".join((
            self.task_type,
            self.target_obj_type,
            self.current_recep_type,
            "" if self.holding is None else str(self.holding),
            "" if self.processed is None else str(self.processed),
            self.processing,
            self.object_category,
            self.receptacle_category,
            ",".join(sorted(self.predicates)),
        ))


@dataclass
class EvidenceState:
    n_pos: int = 0
    n_neg: int = 0
    alpha: float = 1.0
    belief: BeliefState = BeliefState.HYPOTHESIS
    retrieval_count: int = 0
    utility: float = 0.0
    last_episode: int = 0
    recent: list[str] = field(default_factory=list)
    provenance: list[dict[str, Any]] = field(default_factory=list)

    @property
    def total_observable(self) -> int:
        return self.n_pos + self.n_neg

    @property
    def reliability(self) -> float:
        return (self.n_pos + self.alpha) / (self.n_pos + self.n_neg + 2 * self.alpha)

    @property
    def recent_error(self) -> float:
        observable = [x for x in self.recent if x in ("support", "contradict")]
        if not observable:
            return 0.0
        return sum(1 for x in observable if x == "contradict") / len(observable)

    def add_result(self, result: VerificationResult, episode: int,
                   provenance: dict[str, Any] | None = None,
                   window: int = 8):
        if result == VerificationResult.SUPPORT:
            self.n_pos += 1
        elif result == VerificationResult.CONTRADICT:
            self.n_neg += 1
        else:
            return
        self.last_episode = episode
        self.recent.append(result.value)
        if len(self.recent) > window:
            self.recent = self.recent[-window:]
        if provenance:
            self.provenance.append(provenance)
            if len(self.provenance) > 20:
                self.provenance = self.provenance[-20:]


@dataclass
class CausalMemoryUnit:
    condition: Condition
    action: ActionSchema
    effect: Effect
    evidence: EvidenceState = field(default_factory=EvidenceState)
    level: MemoryLevel = MemoryLevel.HYPOTHESIS
    memory_id: str = ""
    source: str = "transition"

    def __post_init__(self):
        if not self.memory_id:
            raw = "||".join((self.condition.key(), self.action.key(), self.effect.key()))
            self.memory_id = sha1(raw.encode("utf-8")).hexdigest()[:16]

    @property
    def reliability(self) -> float:
        return self.evidence.reliability

    @property
    def belief_state(self) -> BeliefState:
        return self.evidence.belief

    def structural_key(self) -> str:
        return "||".join((self.condition.key(), self.action.key(), self.effect.key()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "level": self.level.value,
            "source": self.source,
            "condition": self.condition.__dict__,
            "action": self.action.__dict__,
            "effect": self.effect.__dict__,
            "evidence": {
                "n_pos": self.evidence.n_pos,
                "n_neg": self.evidence.n_neg,
                "reliability": round(self.evidence.reliability, 4),
                "belief": self.evidence.belief.value,
                "retrieval_count": self.evidence.retrieval_count,
                "utility": round(self.evidence.utility, 4),
                "last_episode": self.evidence.last_episode,
                "recent": list(self.evidence.recent),
                "recent_error": round(self.evidence.recent_error, 4),
                "provenance": self.evidence.provenance[-5:],
            },
        }


@dataclass
class TransitionRecord:
    prev_obs: str
    action: str
    next_obs: str
    reward: float = 0.0
    task_type: str = ""
    task_desc: str = ""
    episode: int = 0
    step: int = 0
    context: dict[str, Any] = field(default_factory=dict)

    def provenance(self) -> dict[str, Any]:
        return {
            "episode": self.episode,
            "step": self.step,
            "task_type": self.task_type,
            "action": self.action,
            "reward": self.reward,
        }
