"""
Applicability and effect verification for Seam causal memories.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from memory.extractor import TransitionExtractor
from memory.unit import CausalMemoryUnit, TransitionRecord, VerificationResult

if TYPE_CHECKING:
    from env.adapter import EnvironmentAdapter


class MemoryVerifier:
    def __init__(self, adapter: "EnvironmentAdapter"):
        self.adapter = adapter
        self.extractor = TransitionExtractor(adapter)

    def is_applicable(self, memory: CausalMemoryUnit,
                      transition: TransitionRecord) -> bool:
        action = self.extractor.parse_action_schema(transition.action)
        if not self._action_matches(memory, action):
            return False
        return self.condition_satisfied(memory, transition.context, transition.prev_obs)

    def condition_satisfied(self, memory: CausalMemoryUnit, context: dict,
                            obs: str = "") -> bool:
        cond = memory.condition
        if cond.task_type and context.get("task_type") and cond.task_type != context.get("task_type"):
            return False
        if cond.target_obj_type and context.get("target_obj_type"):
            if cond.target_obj_type != context.get("target_obj_type"):
                return False
        if cond.current_recep_type:
            current = context.get("current_recep_type", "")
            if current and cond.current_recep_type != current:
                return False
        if cond.holding is not None and context.get("holding") is not None:
            if cond.holding != context.get("holding"):
                return False
        if cond.processed is not None and context.get("processed") is not None:
            if cond.processed != context.get("processed"):
                return False
        if cond.processing and context.get("processing"):
            if cond.processing != context.get("processing"):
                return False
        if "closed_receptacle_visible" in cond.predicates and "closed" not in obs.lower():
            return False
        return True

    def verify(self, memory: CausalMemoryUnit,
               transition: TransitionRecord) -> VerificationResult:
        if not self.is_applicable(memory, transition):
            return VerificationResult.UNKNOWN

        parsed_next = self.adapter.parse_observation(transition.next_obs)
        low_next = transition.next_obs.lower()
        effect = memory.effect

        if effect.effect_type == "page_transition":
            next_page = parsed_next.get("page", "")
            if next_page == effect.value:
                return VerificationResult.SUPPORT
            if next_page:
                return VerificationResult.CONTRADICT
            return VerificationResult.UNKNOWN

        if effect.effect_type == "task_complete":
            return (
                VerificationResult.SUPPORT
                if parsed_next.get("task_complete")
                else VerificationResult.UNKNOWN
            )

        if effect.effect_type == "search_failed":
            if parsed_next.get("error"):
                return VerificationResult.SUPPORT
            if parsed_next.get("page") == "search_results":
                return VerificationResult.CONTRADICT
            return VerificationResult.UNKNOWN

        if effect.effect_type == "click_failed":
            if parsed_next.get("error"):
                return VerificationResult.SUPPORT
            if parsed_next.get("page"):
                return VerificationResult.CONTRADICT
            return VerificationResult.UNKNOWN

        if effect.effect_type in ("observe", "observe_target"):
            if self._observation_is_closed_or_failed(parsed_next, low_next):
                return VerificationResult.UNKNOWN
            visible_types = {
                self.adapter.obj_name_to_type(obj)
                for obj in parsed_next.get("objects_here", [])
            }
            visible_types = {v for v in visible_types if v}
            if effect.object_type in visible_types:
                return VerificationResult.SUPPORT
            if "arrived_at" in parsed_next:
                return VerificationResult.CONTRADICT
            return VerificationResult.UNKNOWN

        if effect.effect_type == "opened":
            if "you open" in low_next or " is open" in low_next:
                return VerificationResult.SUPPORT
            if self._is_action_failure(parsed_next, low_next):
                return VerificationResult.CONTRADICT
            return VerificationResult.UNKNOWN

        if effect.effect_type == "holding":
            if parsed_next.get("picked_up"):
                return VerificationResult.SUPPORT
            if self._is_action_failure(parsed_next, low_next):
                return VerificationResult.CONTRADICT
            return VerificationResult.UNKNOWN

        if effect.effect_type == "placed":
            if parsed_next.get("put_down"):
                return VerificationResult.SUPPORT
            if self._is_action_failure(parsed_next, low_next):
                return VerificationResult.CONTRADICT
            return VerificationResult.UNKNOWN

        if effect.effect_type == "processed":
            if parsed_next.get("processed") == effect.value:
                return VerificationResult.SUPPORT
            if self._is_action_failure(parsed_next, low_next):
                return VerificationResult.CONTRADICT
            return VerificationResult.UNKNOWN

        if effect.effect_type == "toggled":
            if parsed_next.get("used_device"):
                return VerificationResult.SUPPORT
            if self._is_action_failure(parsed_next, low_next):
                return VerificationResult.CONTRADICT
            return VerificationResult.UNKNOWN

        if effect.effect_type == "task_complete":
            if "you have completed the task" in low_next:
                return VerificationResult.SUPPORT
            return VerificationResult.UNKNOWN

        return VerificationResult.UNKNOWN

    def _action_matches(self, memory: CausalMemoryUnit, action) -> bool:
        ma = memory.action
        if ma.action_type != action.action_type:
            return False
        if ma.object_type and action.object_type and ma.object_type != action.object_type:
            return False
        if ma.receptacle_type and action.receptacle_type and ma.receptacle_type != action.receptacle_type:
            return False
        if ma.tool_type and action.tool_type and ma.tool_type != action.tool_type:
            return False
        return True

    def _is_action_failure(self, parsed_next: dict, low_next: str) -> bool:
        if parsed_next.get("error"):
            return True
        return any(marker in low_next for marker in (
            "cannot find",
            "can't",
            "nothing happens",
            "not holding",
            "need to",
            "invalid",
        ))

    def _observation_is_closed_or_failed(self, parsed_next: dict, low_next: str) -> bool:
        if self._is_action_failure(parsed_next, low_next):
            return True
        return " is closed" in low_next and "you need to open" not in low_next
