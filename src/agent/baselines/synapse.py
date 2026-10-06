"""Synapse baseline."""

from __future__ import annotations

from typing import Any

from agent.baselines.react import ReActAgent
from agent.baselines.utils import shorten


class SynapseAgent(ReActAgent):
    """Synapse: retrieve successful trajectories as task-specific exemplars."""

    method_name = "synapse"
    exemplar_k = 3

    def _method_instruction(self) -> str:
        return (
            "Use Synapse-style trajectory-as-exemplar prompting: compare the "
            "current task with retrieved successful trajectories, transfer only "
            "the relevant high-level action pattern, and choose one valid action."
        )

    def _initial_memory_context(self) -> str:
        return self._exemplar_context(self._task_obs)

    def act(self, observation: str | None = None, candidates: list[str] | None = None,
            reward: float = 0.0) -> str:
        if observation is not None and len(self._history) > 2:
            self.observe(observation, reward=reward)
            exemplar_context = self._exemplar_context(
                f"{self._task_obs}\n{observation}\n{' '.join(self._actions_taken[-5:])}"
            )
            msg = self._format_step_input(observation, candidates)
            if exemplar_context:
                msg += "\n\n" + exemplar_context
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
                "synapse_exemplars": exemplar_context or None,
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

    def _exemplar_context(self, query: str) -> str:
        if not self.memory:
            return ""
        exemplars = self.memory.retrieve_experiences(
            query, self._task_type, k=self.exemplar_k, successes_only=True
        )
        if not exemplars:
            return ""
        return "[Successful trajectory exemplars]\n" + self._format_exemplars(exemplars)

    def _format_exemplars(self, exemplars: list[dict[str, Any]]) -> str:
        chunks = []
        for exp in exemplars:
            chunks.append(
                f"- [exemplar#{exp.get('id')}, {exp.get('task_type', 'task')}] "
                f"task: {shorten(exp.get('task_obs', ''), 220)}\n"
                f"  trajectory:\n{self._format_exemplar_trajectory(exp)}"
            )
        return "\n".join(chunks)

    @staticmethod
    def _format_exemplar_trajectory(exp: dict[str, Any]) -> str:
        trajectory = exp.get("trajectory") or []
        lines = []
        for idx, entry in enumerate(trajectory[:10]):
            action = shorten(str(entry.get("action", "")), 120)
            observation = shorten(str(entry.get("observation", "")), 180)
            lines.append(f"  {idx + 1}. Action: {action} -> Observation: {observation}")
        if lines:
            return "\n".join(lines)
        actions = " -> ".join(exp.get("actions", [])[:12])
        return f"  actions: {shorten(actions, 600)}"

    def _update_memory_on_episode_end(self, success: bool,
                                      experience: dict[str, Any]) -> dict[str, Any]:
        if not self.memory:
            return {}
        successful = sum(1 for exp in self.memory.experiences if exp.get("success"))
        return {
            "trajectory_memories": len(self.memory.experiences),
            "successful_exemplars": successful,
            "stored_current_trajectory": True,
        }
