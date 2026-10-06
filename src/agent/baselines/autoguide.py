"""AutoGuide baseline."""

from __future__ import annotations

from typing import Any

from agent.baselines.react import ReActAgent
from agent.baselines.utils import parse_prefixed_lines


class AutoGuideAgent(ReActAgent):
    """AutoGuide baseline: generate and retrieve conditional natural-language guides."""

    method_name = "autoguide"

    def reset(self, task_obs: str, task_type: str = ""):
        self._used_guides: list[str] = []
        super().reset(task_obs, task_type=task_type)

    def _method_instruction(self) -> str:
        return (
            "Use context-aware guidelines: apply a guideline only when its IF "
            "condition matches the current task or observation."
        )

    def _initial_memory_context(self) -> str:
        return self._guide_context(self._task_obs)

    def act(self, observation: str | None = None, candidates: list[str] | None = None,
            reward: float = 0.0) -> str:
        if observation is not None and len(self._history) > 2:
            self.observe(observation, reward=reward)
            guide_context = self._guide_context(observation)
            msg = self._format_step_input(observation, candidates)
            if guide_context:
                msg += "\n\n" + guide_context
            self._history.append({"role": "user", "content": msg})

            self._trim_history()
            call_idx_before = self.llm.total_calls
            response = self.llm.chat(self._history, max_tokens=self._action_tokens())
            action = self._parse_candidate_action(response, candidates) if candidates else self._parse_action(response)
            if candidates:
                action = self._validate_action(action, candidates)
            self._actions_taken.append(action)
            self._history.append({"role": "assistant", "content": response.strip() or action})

            self._episode_trace.append({
                "type": "step",
                "step": len(self._actions_taken),
                "observation": observation,
                "guide_context": guide_context or None,
                "candidates_count": len(candidates) if candidates else 0,
                "candidates_sample": candidates[:20] if candidates else [],
                "action": action,
                "llm_calls": [{
                    "call_idx": call_idx_before + 1,
                    "response": response.strip(),
                    "parsed_action": action,
                }],
            })
            return action

        return super().act(observation, candidates=candidates, reward=reward)

    def _guide_context(self, observation: str) -> str:
        if not self.memory:
            return ""
        recent_actions = " ".join(self._actions_taken[-5:])
        query = f"{self._task_obs} {self._task_type} {observation} {recent_actions}"
        guides = self.memory.retrieve_text_items("guide", query, self._task_type, k=8)
        if not guides:
            return ""
        for guide in guides:
            text = guide.get("text", "")
            if text and text not in self._used_guides:
                self._used_guides.append(text)
        return "[Context-aware guidelines]\n" + self.memory.format_text_items(guides)

    def _update_memory_on_episode_end(self, success: bool,
                                      experience: dict[str, Any]) -> dict[str, Any]:
        if not self.memory:
            return {}
        guide_feedback = self.memory.update_guide_feedback(self._used_guides, success)
        result = "succeeded" if success else "failed"
        prompt = [
            {"role": "system", "content": (
                "You generate AutoGuide-style context-aware guidelines from "
                "agent experiences. Output only GUIDE: IF <context> THEN <advice> "
                "lines. Keep each guideline concise and action-oriented."
            )},
            {"role": "user", "content": (
                f"Task type: {self._task_type}\n"
                f"Task: {self._task_obs}\n"
                f"Result: {result}\n"
                f"Trajectory:\n{self._trajectory_text()}\n\n"
                "Write up to 4 context-aware guidelines."
            )},
        ]
        response = self.llm.chat(prompt, max_tokens=512 if self.llm.thinking_disabled else 1024)
        lines = parse_prefixed_lines(response, ("guide",), limit=4)
        added = self.memory.add_text_items("guide", lines, self.episode, self._task_type)
        self._episode_trace.append({
            "type": "memory_update",
            "kind": "guide",
            "response": response,
            "added": added,
            "guide_feedback_updated": guide_feedback,
        })
        return {"guides_added": len(added), "guide_feedback_updated": guide_feedback}
