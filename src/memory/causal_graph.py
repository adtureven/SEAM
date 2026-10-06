"""
Self-evolving condition-action-effect memory graph for Seam.

The public class name remains ``HierarchicalCausalGraph`` for compatibility
with existing experiment scripts, but the internal representation follows the
paper method: M0 hypothesis memories, M1 stable memories, M2 abstract memories,
and bounded retrieval over condition-action-effect units.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from memory.extractor import TransitionExtractor
from memory.unit import (
    ActionSchema,
    BeliefState,
    CausalMemoryUnit,
    Condition,
    Effect,
    EvidenceState,
    MemoryLevel,
    TransitionRecord,
    VerificationResult,
)
from memory.verifier import MemoryVerifier

if TYPE_CHECKING:
    from env.adapter import EnvironmentAdapter


@dataclass
class CausalChain:
    task_type: str
    steps: list[str]
    confirmations: int = 0
    belief_state: BeliefState = BeliefState.HYPOTHESIS


@dataclass
class RetrievedMemory:
    memory: CausalMemoryUnit
    score: float
    predicted_effect: str


class HierarchicalCausalGraph:
    """Multi-level causal memory graph with evidence-based retrieval."""

    def __init__(self, stability_threshold: int = 4, aggregation_threshold: int = 2,
                 adapter: "EnvironmentAdapter | None" = None,
                 stable_reliability: float = 0.7,
                 revision_error_threshold: float = 0.5):
        self._stability_threshold = stability_threshold
        self._aggregation_threshold = aggregation_threshold
        self._stable_reliability = stable_reliability
        self._revision_error_threshold = revision_error_threshold
        self.adapter = adapter
        self.extractor = TransitionExtractor(adapter) if adapter else None
        self.verifier = MemoryVerifier(adapter) if adapter else None

        self._memories: dict[str, CausalMemoryUnit] = {}
        self._structural_index: dict[str, str] = {}
        self._action_index: dict[str, set[str]] = defaultdict(set)
        self._task_index: dict[str, set[str]] = defaultdict(set)
        self._effect_index: dict[str, set[str]] = defaultdict(set)
        self._causal_chains: dict[str, CausalChain] = {}
        self._transition_count = 0
        self._verification_count = 0
        self._unknown_count = 0
        self._pruned_count = 0
        self._abstract_count = 0

    # ------------------------------------------------------------------
    # Core memory operations
    # ------------------------------------------------------------------

    def add_hypothesis(self, memory: CausalMemoryUnit, episode: int = 0,
                       provenance: dict[str, Any] | None = None,
                       count_as_support: bool = True) -> CausalMemoryUnit:
        key = memory.structural_key()
        if key in self._structural_index:
            existing = self._memories[self._structural_index[key]]
            if count_as_support:
                existing.evidence.add_result(
                    VerificationResult.SUPPORT,
                    episode,
                    provenance=provenance,
                )
                self._update_belief(existing)
            elif provenance:
                existing.evidence.provenance.append(provenance)
            return existing

        memory.level = MemoryLevel.HYPOTHESIS
        memory.evidence.belief = BeliefState.HYPOTHESIS
        if provenance:
            memory.evidence.provenance.append(provenance)
        if count_as_support and memory.source.startswith("transition"):
            memory.evidence.add_result(
                VerificationResult.SUPPORT,
                episode,
                provenance=provenance,
            )
        self._memories[memory.memory_id] = memory
        self._structural_index[key] = memory.memory_id
        self._index_memory(memory)
        return memory

    def verify_transition(self, transition: TransitionRecord) -> dict[str, int]:
        """Verify applicable memories against an observed transition."""
        if not self.verifier:
            return {"support": 0, "contradict": 0, "unknown": 0}

        counts = {"support": 0, "contradict": 0, "unknown": 0}
        self._transition_count += 1
        for memory in list(self._memories.values()):
            result = self.verifier.verify(memory, transition)
            if result == VerificationResult.UNKNOWN:
                counts["unknown"] += 1
                self._unknown_count += 1
                continue
            memory.evidence.add_result(result, transition.episode, transition.provenance())
            self._verification_count += 1
            counts[result.value] += 1
            self._update_belief(memory)
        return counts

    def ingest_transition(self, transition: TransitionRecord) -> dict[str, int]:
        """Extract candidate memories and verify existing ones from one transition."""
        stats = self.verify_transition(transition)
        extracted = 0
        if self.extractor:
            for memory in self.extractor.extract(transition):
                self.add_hypothesis(memory, transition.episode, transition.provenance())
                extracted += 1
        stats["extracted"] = extracted
        return stats

    def _update_belief(self, memory: CausalMemoryUnit):
        ev = memory.evidence
        if ev.belief == BeliefState.HYPOTHESIS:
            if ev.total_observable >= self._stability_threshold and ev.reliability >= self._stable_reliability:
                ev.belief = BeliefState.STABLE
                if memory.level == MemoryLevel.HYPOTHESIS:
                    memory.level = MemoryLevel.STABLE
        elif ev.belief == BeliefState.STABLE:
            if ev.recent_error >= self._revision_error_threshold and ev.n_neg >= 2:
                ev.belief = BeliefState.REVISING
        elif ev.belief == BeliefState.REVISING:
            if ev.total_observable >= self._stability_threshold and ev.reliability >= self._stable_reliability:
                ev.belief = BeliefState.STABLE
            elif ev.reliability < 0.3 and ev.n_neg >= self._stability_threshold:
                memory.level = MemoryLevel.HYPOTHESIS

    def _index_memory(self, memory: CausalMemoryUnit):
        self._action_index[memory.action.action_type].add(memory.memory_id)
        if memory.condition.task_type:
            self._task_index[memory.condition.task_type].add(memory.memory_id)
        self._effect_index[memory.effect.effect_type].add(memory.memory_id)

    def _remove_memory(self, memory_id: str):
        memory = self._memories.pop(memory_id, None)
        if not memory:
            return
        self._structural_index.pop(memory.structural_key(), None)
        for index in (self._action_index, self._task_index, self._effect_index):
            for ids in index.values():
                ids.discard(memory_id)
        self._pruned_count += 1

    # ------------------------------------------------------------------
    # Retrieval and prediction
    # ------------------------------------------------------------------

    def retrieve(self, context: dict[str, Any], candidates: list[str] | None = None,
                 top_k: int = 8) -> list[RetrievedMemory]:
        if not self.adapter or not self.verifier:
            return []
        candidate_actions = candidates or []
        action_schemas = [self.extractor.parse_action_schema(a) for a in candidate_actions] if self.extractor else []

        ids: set[str] = set()
        task_type = context.get("task_type", "")
        if task_type and task_type in self._task_index:
            ids |= self._task_index[task_type]
        if action_schemas:
            for schema in action_schemas:
                ids |= self._action_index.get(schema.action_type, set())
        if not ids:
            ids = set(self._memories)

        retrieved: list[RetrievedMemory] = []
        for memory_id in ids:
            memory = self._memories[memory_id]
            if memory.evidence.belief != BeliefState.STABLE:
                continue
            sat = self._condition_score(memory, context)
            if sat <= 0:
                continue
            rel = self._relevance_score(memory, context)
            q = memory.reliability
            score = rel * q * sat
            if memory.level == MemoryLevel.ABSTRACT:
                score *= 0.85
            if candidates and action_schemas:
                action_match = max(self._action_match_score(memory.action, a) for a in action_schemas)
                score *= action_match
            if score > 0.05:
                memory.evidence.retrieval_count += 1
                retrieved.append(RetrievedMemory(memory, score, self.format_effect(memory)))

        retrieved.sort(key=lambda x: -x.score)
        return retrieved[:top_k]

    def score_action(self, action: str, context: dict[str, Any]) -> float:
        if not self.extractor:
            return 0.0
        schema = self.extractor.parse_action_schema(action)
        best = 0.0
        for item in self.retrieve(context, candidates=[action], top_k=6):
            match = self._action_match_score(item.memory.action, schema)
            if match <= 0:
                continue
            best = max(best, item.score * match)
        return best

    def predict_effects(self, action: str, context: dict[str, Any],
                        top_k: int = 3) -> list[RetrievedMemory]:
        return self.retrieve(context, candidates=[action], top_k=top_k)

    def format_hint(self, context: dict[str, Any], candidates: list[str] | None = None,
                    top_k: int = 5) -> str:
        retrieved = self.retrieve(context, candidates=candidates, top_k=top_k)
        if not retrieved:
            return ""
        lines = []
        for item in retrieved:
            memory = item.memory
            strength = "stable" if memory.evidence.belief == BeliefState.STABLE else "hypothesis"
            lines.append(
                f"- If action matches `{self._action_text(memory.action)}`, "
                f"expect {item.predicted_effect} "
                f"(q={memory.reliability:.2f}, {strength})"
            )
        return "\n[Memory]\n" + "\n".join(lines)

    def _condition_score(self, memory: CausalMemoryUnit, context: dict[str, Any]) -> float:
        cond = memory.condition
        score = 1.0
        if cond.task_type:
            if context.get("task_type") == cond.task_type:
                score *= 1.0
            else:
                score *= 0.4
        if cond.target_obj_type:
            if context.get("target_obj_type") == cond.target_obj_type:
                score *= 1.0
            else:
                score *= 0.5
        if cond.current_recep_type:
            current = context.get("current_recep_type", "")
            if current == cond.current_recep_type:
                score *= 1.0
            elif not current:
                score *= 0.6
            else:
                score *= 0.2
        if cond.holding is not None and context.get("holding") is not None:
            score *= 1.0 if cond.holding == context.get("holding") else 0.2
        if cond.processed is not None and context.get("processed") is not None:
            score *= 1.0 if cond.processed == context.get("processed") else 0.4
        return score

    def _relevance_score(self, memory: CausalMemoryUnit, context: dict[str, Any]) -> float:
        rel = 0.5
        target = context.get("target_obj_type", "")
        processing = context.get("processing", "")
        if target and (memory.action.object_type == target or memory.effect.object_type == target
                       or memory.condition.target_obj_type == target):
            rel += 0.3
        if processing and (memory.action.action_type == processing or memory.effect.value == processing):
            rel += 0.2
        if memory.effect.effect_type in ("task_complete", "observe_target"):
            rel += 0.1
        return min(rel, 1.0)

    def _action_match_score(self, memory_action: ActionSchema, action: ActionSchema) -> float:
        if memory_action.action_type != action.action_type:
            return 0.0
        score = 1.0
        if memory_action.object_type and action.object_type:
            score *= 1.0 if memory_action.object_type == action.object_type else 0.5
        if memory_action.receptacle_type and action.receptacle_type:
            score *= 1.0 if memory_action.receptacle_type == action.receptacle_type else 0.4
        if memory_action.tool_type and action.tool_type:
            score *= 1.0 if memory_action.tool_type == action.tool_type else 0.4
        return score

    def _action_text(self, action: ActionSchema) -> str:
        parts = [action.action_type]
        if action.object_type:
            parts.append(action.object_type)
        if action.receptacle_type:
            parts.append(action.receptacle_type)
        if action.tool_type:
            parts.append(action.tool_type)
        return " ".join(parts)

    def format_effect(self, memory: CausalMemoryUnit) -> str:
        effect = memory.effect
        if effect.text:
            return effect.text
        if effect.effect_type == "observe":
            return f"{effect.object_type} becomes visible at {effect.receptacle_type}"
        if effect.effect_type == "holding":
            return f"agent holds {effect.object_type}"
        if effect.effect_type == "placed":
            return f"{effect.object_type} is placed at {effect.receptacle_type}"
        if effect.effect_type == "processed":
            return f"{effect.object_type} becomes {effect.value}"
        return effect.effect_type

    # ------------------------------------------------------------------
    # Evolution operations
    # ------------------------------------------------------------------

    def abstract_stable_memories(self, episode: int,
                                 min_members: int = 2,
                                 min_reliability: float = 0.6) -> int:
        if not self.adapter:
            return 0
        groups: dict[tuple[str, str, str, str], list[CausalMemoryUnit]] = defaultdict(list)
        for memory in self._memories.values():
            if memory.level != MemoryLevel.STABLE or memory.evidence.belief != BeliefState.STABLE:
                continue
            obj_cat = self.adapter.get_object_category(
                memory.action.object_type or memory.effect.object_type
            ) or ""
            recep_cat = self.adapter.get_receptacle_category(
                memory.action.receptacle_type or memory.effect.receptacle_type
            ) or ""
            key = (memory.action.action_type, memory.effect.effect_type,
                   memory.effect.value, obj_cat or memory.action.object_type)
            if obj_cat or recep_cat:
                groups[key + (recep_cat,)].append(memory)

        created = 0
        for key, members in groups.items():
            if len(members) < min_members:
                continue
            avg_rel = sum(m.reliability for m in members) / len(members)
            if avg_rel < min_reliability:
                continue
            action_type, effect_type, effect_value, obj_cat, recep_cat = key
            cond = Condition(
                task_type="",
                object_category=obj_cat,
                receptacle_category=recep_cat,
                predicates=("abstracted",),
            )
            action = ActionSchema(action_type, raw=f"{action_type}<{obj_cat},{recep_cat}>")
            effect = Effect(effect_type, value=effect_value, text=f"abstract {effect_type} for {obj_cat}")
            abstract = CausalMemoryUnit(
                condition=cond,
                action=action,
                effect=effect,
                evidence=EvidenceState(),
                level=MemoryLevel.ABSTRACT,
                source="abstraction",
            )
            abstract.evidence.belief = BeliefState.HYPOTHESIS
            if abstract.structural_key() in self._structural_index:
                continue
            existing = self.add_hypothesis(abstract, episode, {
                "episode": episode,
                "source": "abstraction",
                "members": [m.memory_id for m in members[:8]],
            }, count_as_support=False)
            existing.level = MemoryLevel.ABSTRACT
            created += 1
        self._abstract_count += created
        return created

    def revise_and_prune(self, episode: int, staleness: int = 20,
                         min_reliability: float = 0.25) -> int:
        removed = 0
        for memory_id, memory in list(self._memories.items()):
            ev = memory.evidence
            if ev.belief == BeliefState.STABLE:
                continue
            if ev.recent_error >= self._revision_error_threshold and ev.n_neg >= 2:
                ev.belief = BeliefState.REVISING
            stale = episode - ev.last_episode > staleness
            low_quality = ev.total_observable >= self._stability_threshold and ev.reliability < min_reliability
            low_usage = ev.retrieval_count == 0
            if (stale and low_quality) or (low_quality and low_usage):
                self._remove_memory(memory_id)
                removed += 1
        return removed

    # ------------------------------------------------------------------
    # Compatibility helpers used by older code paths
    # ------------------------------------------------------------------

    def observe(self, obj_type: str, recep_type: str, episode: int):
        memory = CausalMemoryUnit(
            condition=Condition(target_obj_type=obj_type),
            action=ActionSchema("goto", receptacle_type=recep_type),
            effect=Effect("observe", object_type=obj_type, receptacle_type=recep_type,
                          text=f"{obj_type} becomes visible at {recep_type}"),
            source="compat_observe",
        )
        self.add_hypothesis(memory, episode, {"episode": episode, "source": "observe"})

    def observe_absence(self, obj_type: str, recep_type: str, episode: int):
        for memory in self._memories.values():
            if (memory.action.action_type == "goto"
                    and memory.action.receptacle_type == recep_type
                    and memory.effect.object_type == obj_type):
                memory.evidence.add_result(VerificationResult.CONTRADICT, episode, {
                    "episode": episode,
                    "source": "absence",
                })
                self._update_belief(memory)

    def observe_action_effect(self, action_type: str, appliance_type: str, episode: int):
        memory = CausalMemoryUnit(
            condition=Condition(processing=action_type),
            action=ActionSchema(action_type, tool_type=appliance_type),
            effect=Effect("processed", receptacle_type=appliance_type, value=action_type,
                          text=f"object becomes {action_type}"),
            source="compat_action_effect",
        )
        self.add_hypothesis(memory, episode, {"episode": episode, "source": "action_effect"})

    def query_location(self, obj_type: str, top_k: int = 5) -> list[tuple[str, float, BeliefState]]:
        results = []
        for memory in self._memories.values():
            if memory.evidence.belief != BeliefState.STABLE:
                continue
            if memory.effect.effect_type not in ("observe", "observe_target"):
                continue
            if memory.effect.object_type != obj_type:
                continue
            recep = memory.effect.receptacle_type or memory.action.receptacle_type
            if not recep:
                continue
            results.append((recep, memory.reliability, memory.evidence.belief))
        results.sort(key=lambda x: (-int(x[2] == BeliefState.STABLE), -x[1]))
        return results[:top_k]

    def query_appliance(self, action_type: str) -> list[tuple[str, float, BeliefState]]:
        results = []
        for memory in self._memories.values():
            if memory.evidence.belief != BeliefState.STABLE:
                continue
            if memory.action.action_type == action_type and memory.effect.effect_type == "processed":
                tool = memory.action.tool_type or memory.effect.receptacle_type
                if tool:
                    results.append((tool, memory.reliability, memory.evidence.belief))
        results.sort(key=lambda x: (-int(x[2] == BeliefState.STABLE), -x[1]))
        return results

    def query_next_action(self, task_state: dict) -> tuple[str, str, float] | None:
        context = {
            "task_type": task_state.get("goal", ""),
            "holding": task_state.get("holding"),
            "processed": task_state.get("processed"),
            "processing": task_state.get("processing", ""),
            "target_obj_type": task_state.get("target_obj_type", ""),
        }
        best = None
        for memory in self.retrieve(context, top_k=3):
            action = memory.memory.action
            target = action.receptacle_type or action.tool_type or action.object_type
            candidate = (action.action_type, target, memory.score)
            if best is None or candidate[2] > best[2]:
                best = candidate
        return best

    def record_causal_chain(self, task_type: str, steps: list[str], success: bool, episode: int):
        if not success:
            return
        if task_type in self._causal_chains:
            chain = self._causal_chains[task_type]
            if chain.steps == steps:
                chain.confirmations += 1
            elif chain.belief_state != BeliefState.STABLE and len(steps) < len(chain.steps):
                chain.steps = steps
                chain.confirmations = 1
        else:
            chain = CausalChain(task_type, steps, confirmations=1)
            self._causal_chains[task_type] = chain
        chain = self._causal_chains[task_type]
        if chain.confirmations >= self._stability_threshold:
            chain.belief_state = BeliefState.STABLE

    def query_procedure(self, task_type: str) -> CausalChain | None:
        chain = self._causal_chains.get(task_type)
        if chain and chain.confirmations >= 2:
            return chain
        return None

    def get_location_hint(self, obj_type: str) -> str:
        locations = self.query_location(obj_type, top_k=4)
        if not locations:
            return ""
        parts = []
        for recep_type, conf, belief in locations:
            name = recep_type.replace("Type", "").lower()
            label = "stable" if belief == BeliefState.STABLE else f"{conf:.0%}"
            parts.append(f"{name} ({label})")
        obj_name = obj_type.replace("Type", "").lower()
        return f"Search for {obj_name}: {', '.join(parts)}"

    def get_procedure_hint(self, task_type: str) -> str:
        chain = self.query_procedure(task_type)
        if not chain:
            return ""
        strength = "Proven" if chain.belief_state == BeliefState.STABLE else "Likely"
        return f"{strength} procedure: {' -> '.join(chain.steps[:8])}"

    def prune(self, current_episode: int, staleness: int = 15, min_confidence: float = 0.3) -> int:
        return self.revise_and_prune(current_episode, staleness, min_confidence)

    def discover_conditions(self, current_episode: int) -> int:
        changed = 0
        for memory in self._memories.values():
            if memory.evidence.belief == BeliefState.STABLE and memory.evidence.recent_error >= self._revision_error_threshold:
                memory.evidence.belief = BeliefState.REVISING
                changed += 1
        return changed

    def get_unabsorbed_instances(self):
        return []

    def promote_to_l1(self, obj_type: str, recep_type: str, count: int, episode: int):
        for _ in range(max(1, count)):
            self.observe(obj_type, recep_type, episode)

    def add_l2_rule(self, source_cat: str, target_cat: str, relation: str,
                    confidence: float, episode: int):
        memory = CausalMemoryUnit(
            condition=Condition(object_category=source_cat, receptacle_category=target_cat,
                                predicates=("abstracted",)),
            action=ActionSchema("goto", raw=f"goto<{target_cat}>"),
            effect=Effect(relation, value=target_cat, text=f"{source_cat} -> {target_cat}"),
            evidence=EvidenceState(n_pos=max(1, int(confidence * 2)), n_neg=0),
            level=MemoryLevel.ABSTRACT,
            source="compat_l2",
        )
        memory.evidence.belief = BeliefState.HYPOTHESIS
        self.add_hypothesis(memory, episode, {"episode": episode, "source": "l2_rule"})

    def mark_instances_absorbed(self, obj_type: str, recep_type: str):
        return None

    # ------------------------------------------------------------------
    # Stats and serialization
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        m0 = sum(1 for m in self._memories.values() if m.level == MemoryLevel.HYPOTHESIS)
        m1 = sum(1 for m in self._memories.values() if m.level == MemoryLevel.STABLE)
        m2 = sum(1 for m in self._memories.values() if m.level == MemoryLevel.ABSTRACT)
        m0_stable = sum(1 for m in self._memories.values()
                        if m.level == MemoryLevel.HYPOTHESIS and m.evidence.belief == BeliefState.STABLE)
        m1_stable = sum(1 for m in self._memories.values()
                        if m.level == MemoryLevel.STABLE and m.evidence.belief == BeliefState.STABLE)
        m2_stable = sum(1 for m in self._memories.values()
                        if m.level == MemoryLevel.ABSTRACT and m.evidence.belief == BeliefState.STABLE)
        stable = m0_stable + m1_stable + m2_stable
        revising = sum(1 for m in self._memories.values() if m.evidence.belief == BeliefState.REVISING)
        by_effect = defaultdict(int)
        by_action = defaultdict(int)
        for memory in self._memories.values():
            by_effect[memory.effect.effect_type] += 1
            by_action[memory.action.action_type] += 1
        chains_stable = sum(1 for c in self._causal_chains.values() if c.belief_state == BeliefState.STABLE)
        return {
            "memory_units": len(self._memories),
            "m0_hypothesis": m0,
            "m1_stable": m1,
            "m2_abstract": m2,
            "m0_belief_stable": m0_stable,
            "m1_belief_stable": m1_stable,
            "m2_belief_stable": m2_stable,
            "belief_stable": stable,
            "belief_revising": revising,
            "by_effect": dict(by_effect),
            "by_action": dict(by_action),
            "causal_chains": len(self._causal_chains),
            "chains_stable": chains_stable,
            "transitions_seen": self._transition_count,
            "verifications": self._verification_count,
            "unknown_verifications": self._unknown_count,
            "pruned": self._pruned_count,
            "abstract_created": self._abstract_count,
            # Backward-compatible names used by old printers/results.
            "l0_instances": m0,
            "l0_unabsorbed": m0,
            "l1_edges": m1,
            "l1_stable": m1_stable,
            "l1_exploring": m0,
            "l1_located_at": by_effect.get("observe", 0) + by_effect.get("observe_target", 0),
            "l1_processed_at": by_effect.get("processed", 0),
            "l1_conditional": revising,
            "l2_rules": m2,
            "total_observations": self._transition_count,
            "total_surprises": sum(m.evidence.n_neg for m in self._memories.values()),
        }

    def to_dict(self) -> dict:
        return {
            "stats": self.get_stats(),
            "memories": [m.to_dict() for m in self._memories.values()],
            "causal_chains": [
                {
                    "task_type": c.task_type,
                    "steps": c.steps,
                    "confirmations": c.confirmations,
                    "belief_state": c.belief_state.value,
                }
                for c in self._causal_chains.values()
            ],
        }
