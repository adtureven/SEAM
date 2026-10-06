"""Reflexion baseline."""

from __future__ import annotations

from typing import Any

from agent.baselines.react import ReActAgent
from agent.baselines.utils import parse_prefixed_lines


class ReflexionAgent(ReActAgent):
    """Reflexion baseline: store verbal reflections in episodic memory."""

    method_name = "reflexion"

    def _method_instruction(self) -> str:
        return (
            "Use ReAct for acting. If reflections are provided, treat them as "
            "lessons from prior trials and apply only the relevant ones."
        )

    def _initial_memory_context(self) -> str:
        if not self.memory:
            return ""
        query = f"{self._task_obs} {self._task_type}"
        reflections = self.memory.retrieve_text_items("reflection", query, self._task_type, k=3)
        if not reflections:
            return ""
        return "[Reflections]\n" + self.memory.format_text_items(reflections)

    def _update_memory_on_episode_end(self, success: bool,
                                      experience: dict[str, Any]) -> dict[str, Any]:
        if not self.memory:
            return {}
        if success:
            return {"reflections_added": 0, "reason": "success"}
        result = "succeeded" if success else "failed"
        prompt = [
            {"role": "system", "content": (
                "You are the Reflexion evaluator and self-reflection module for "
                "a language agent. Given a failed trajectory, diagnose the mistake "
                "and write concise verbal feedback for the next trial. Output only "
                "REFLECTION: lines. Do not include action commands."
            )},
            {"role": "user", "content": (
                f"Task type: {self._task_type}\n"
                f"Task: {self._task_obs}\n"
                f"Result: {result}\n"
                f"Trajectory:\n{self._trajectory_text()}\n\n"
                "Write up to 3 generally useful reflections."
            )},
        ]
        response = self.llm.chat(prompt, max_tokens=512 if self.llm.thinking_disabled else 1024)
        lines = parse_prefixed_lines(response, ("reflection",), limit=3)
        added = self.memory.add_text_items("reflection", lines, self.episode, self._task_type)
        self._episode_trace.append({
            "type": "memory_update",
            "kind": "reflection",
            "response": response,
            "added": added,
        })
        return {"reflections_added": len(added)}
