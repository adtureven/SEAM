"""
Evidence-based memory evolution for Seam.

The engine is the slow process in the paper method. Online action selection only
retrieves bounded memory hints; this engine consumes stored transitions at
episode boundaries and updates the causal memory graph.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from memory.causal_graph import HierarchicalCausalGraph
from memory.unit import TransitionRecord

if TYPE_CHECKING:
    from env.adapter import EnvironmentAdapter


class EvolutionEngine:
    def __init__(self, graph: HierarchicalCausalGraph,
                 adapter: "EnvironmentAdapter | None" = None,
                 aggregation_threshold: int = 2,
                 abstraction_min_members: int = 2,
                 abstraction_min_confidence: float = 0.6,
                 revision_error_threshold: float = 0.5,
                 enable_self_evolution: bool = True,
                 enable_extraction: bool = True,
                 enable_verification: bool = True,
                 enable_abstraction: bool = True,
                 enable_revision: bool = True,
                 enable_pruning: bool = True):
        self.graph = graph
        self.adapter = adapter
        self._aggregation_threshold = aggregation_threshold
        self._abstraction_min_members = abstraction_min_members
        self._abstraction_min_confidence = abstraction_min_confidence
        self._revision_error_threshold = revision_error_threshold
        self.enable_self_evolution = enable_self_evolution
        self.enable_extraction = enable_extraction
        self.enable_verification = enable_verification
        self.enable_abstraction = enable_abstraction
        self.enable_revision = enable_revision
        self.enable_pruning = enable_pruning
        self._transition_buffer: list[TransitionRecord] = []
        self._episode_trajectories: list[dict] = []
        self._evolution_log: list[dict] = []

    def record_transition(self, transition: TransitionRecord):
        if not self.enable_self_evolution:
            return
        self._transition_buffer.append(transition)

    def on_episode_end(self, task_type: str, steps: list[str],
                       success: bool, episode: int):
        if not self.enable_self_evolution:
            self._log("self_evolution_disabled", episode, {
                "task_type": task_type,
                "success": success,
                "num_steps": len(steps),
            })
            return

        self._episode_trajectories.append({
            "task_type": task_type,
            "steps": steps,
            "success": success,
            "episode": episode,
            "num_transitions": len(self._transition_buffer),
        })

        if self._transition_buffer:
            self._process_transition_buffer(episode)

        self._evolve_structures(episode)

    def _process_transition_buffer(self, episode: int):
        totals = defaultdict(int)
        for transition in self._transition_buffer:
            if self.enable_verification and self.enable_extraction:
                stats = self.graph.ingest_transition(transition)
            elif self.enable_extraction:
                extracted = 0
                if self.graph.extractor:
                    for memory in self.graph.extractor.extract(transition):
                        self.graph.add_hypothesis(
                            memory,
                            transition.episode,
                            transition.provenance(),
                        )
                        extracted += 1
                stats = {
                    "support": 0,
                    "contradict": 0,
                    "unknown": 0,
                    "extracted": extracted,
                }
            elif self.enable_verification:
                stats = self.graph.verify_transition(transition)
                stats["extracted"] = 0
            else:
                stats = {"support": 0, "contradict": 0, "unknown": 0, "extracted": 0}
            for key, value in stats.items():
                totals[key] += value

        self._log("transition_evolution", episode, {
            "num_transitions": len(self._transition_buffer),
            "support": totals["support"],
            "contradict": totals["contradict"],
            "unknown": totals["unknown"],
            "extracted": totals["extracted"],
        })
        self._transition_buffer.clear()

    def _evolve_structures(self, episode: int):
        if self.enable_abstraction:
            created = self.graph.abstract_stable_memories(
                episode,
                min_members=self._abstraction_min_members,
                min_reliability=self._abstraction_min_confidence,
            )
            if created:
                self._log("abstraction", episode, {"created": created})

        if self.enable_revision or self.enable_pruning:
            removed = self.graph.revise_and_prune(
                episode,
                staleness=20,
                min_reliability=0.25,
            ) if self.enable_pruning else 0
            revised = self.graph.discover_conditions(episode) if self.enable_revision else 0
            if revised:
                self._log("revision", episode, {"marked_revising": revised})
            if removed:
                self._log("pruning", episode, {"removed": removed})

    def _learn_action_effects_from_steps(self, steps: list[str], episode: int):
        if not self.adapter:
            return
        learned = 0
        for step in steps:
            effect = self.adapter.extract_action_effect(step)
            if effect:
                proc_type, appliance_name = effect
                app_type = self.adapter.infer_appliance_type(appliance_name)
                if app_type:
                    self.graph.observe_action_effect(proc_type, app_type, episode)
                    learned += 1
        if learned:
            self._log("successful_action_effects", episode, {"learned": learned})

    def _log(self, evo_type: str, episode: int, details: dict):
        self._evolution_log.append({
            "type": evo_type,
            "episode": episode,
            "details": details,
        })

    def get_stats(self) -> dict:
        type_counts = defaultdict(int)
        for entry in self._evolution_log:
            type_counts[entry["type"]] += 1
        return {
            "total_evolutions": len(self._evolution_log),
            "by_type": dict(type_counts),
            "trajectories_recorded": len(self._episode_trajectories),
            "buffered_transitions": len(self._transition_buffer),
        }
