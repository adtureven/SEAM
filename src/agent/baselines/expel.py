"""ExpeL baseline."""

from __future__ import annotations

from typing import Any

from agent.baselines.react import ReActAgent
from agent.baselines.utils import parse_prefixed_lines


class ExpeLAgent(ReActAgent):
    """ExpeL baseline: extract reusable insights from accumulated experiences."""

    method_name = "expel"

    def _method_instruction(self) -> str:
        return (
            "Use experiential learning: apply relevant extracted insights and "
            "similar prior experiences, then choose one action."
        )

    def _initial_memory_context(self) -> str:
        if not self.memory:
            return ""
        query = f"{self._task_obs} {self._task_type}"
        insights = self.memory.retrieve_text_items("insight", query, self._task_type, k=8)
        experiences = self.memory.retrieve_experiences(query, self._task_type, k=2)
        parts = []
        if insights:
            parts.append("[Insights]\n" + self.memory.format_text_items(insights))
        if experiences:
            parts.append("[Similar experiences]\n" + self.memory.format_experiences(experiences))
        return "\n\n".join(parts)

    def _update_memory_on_episode_end(self, success: bool,
                                      experience: dict[str, Any]) -> dict[str, Any]:
        if not self.memory:
            return {}
        prior = self.memory.retrieve_text_items(
            "insight", f"{self._task_obs} {self._task_type}", self._task_type, k=8
        )
        prior_text = self.memory.format_text_items(prior) if prior else "(none)"
        similar = self.memory.retrieve_experiences(
            f"{self._task_obs} {self._task_type}", self._task_type, k=6
        )
        success_cases = [exp for exp in similar if exp.get("success")][:3]
        failure_cases = [exp for exp in similar if not exp.get("success")][:3]
        case_text = []
        if success_cases:
            case_text.append("[Successful experiences]\n" + self.memory.format_experiences(success_cases))
        if failure_cases:
            case_text.append("[Failed experiences]\n" + self.memory.format_experiences(failure_cases))
        experience_text = "\n\n".join(case_text) if case_text else "(none)"
        result = "succeeded" if success else "failed"
        prompt = [
            {"role": "system", "content": (
                "You implement ExpeL-style experiential learning. Compare "
                "successful and failed experiences, keep useful prior insights, "
                "and extract updated reusable insights. Output only INSIGHT: "
                "lines. Each insight must be general enough for future tasks."
            )},
            {"role": "user", "content": (
                f"Existing insights:\n{prior_text}\n\n"
                f"Related experience pool:\n{experience_text}\n\n"
                f"Task type: {self._task_type}\n"
                f"Task: {self._task_obs}\n"
                f"Result: {result}\n"
                f"Trajectory:\n{self._trajectory_text()}\n\n"
                "Write up to 4 new or revised insights."
            )},
        ]
        response = self.llm.chat(prompt, max_tokens=512 if self.llm.thinking_disabled else 1024)
        lines = parse_prefixed_lines(response, ("insight",), limit=4)
        added = self.memory.add_text_items("insight", lines, self.episode, self._task_type)
        self._episode_trace.append({
            "type": "memory_update",
            "kind": "insight",
            "response": response,
            "added": added,
        })
        return {"insights_added": len(added)}
