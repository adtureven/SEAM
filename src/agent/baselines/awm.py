"""Agent Workflow Memory baseline."""

from __future__ import annotations

from typing import Any

from agent.baselines.react import ReActAgent
from agent.baselines.utils import parse_json_object, shorten


class AWMAgent(ReActAgent):
    """Agent Workflow Memory: induce, integrate, and reuse successful workflows."""

    method_name = "awm"

    def reset(self, task_obs: str, task_type: str = ""):
        self._used_workflow_ids: list[int] = []
        super().reset(task_obs, task_type=task_type)

    def _method_instruction(self) -> str:
        return (
            "Use Agent Workflow Memory: if a retrieved workflow matches the task, "
            "follow its abstract procedure while adapting every step to the current "
            "observation and valid action candidates."
        )

    def _initial_memory_context(self) -> str:
        return self._workflow_context(self._task_obs)

    def act(self, observation: str | None = None, candidates: list[str] | None = None,
            reward: float = 0.0) -> str:
        if observation is not None and len(self._history) > 2:
            self.observe(observation, reward=reward)
            workflow_context = self._workflow_context(
                f"{self._task_obs}\n{observation}\n{' '.join(self._actions_taken[-5:])}"
            )
            msg = self._format_step_input(observation, candidates)
            if workflow_context:
                msg += "\n\n" + workflow_context
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
                "workflow_context": workflow_context or None,
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

    def _workflow_context(self, query: str) -> str:
        if not self.memory:
            return ""
        workflows = self.memory.retrieve_workflows(query, self._task_type, k=3)
        if not workflows:
            return ""
        for workflow in workflows:
            workflow_id = workflow.get("id")
            if workflow_id is not None and workflow_id not in self._used_workflow_ids:
                self._used_workflow_ids.append(workflow_id)
        return "[Retrieved workflows]\n" + self.memory.format_workflows(workflows)

    def _update_memory_on_episode_end(self, success: bool,
                                      experience: dict[str, Any]) -> dict[str, Any]:
        if not self.memory:
            return {}
        feedback_updated = self.memory.update_workflow_feedback(self._used_workflow_ids, success)
        if not success:
            return {
                "workflow_feedback_updated": feedback_updated,
                "workflow_added": 0,
                "workflow_updated": 0,
                "reason": "AWM induces workflows only from successful trajectories",
            }

        query = f"{self._task_obs} {self._task_type} {' '.join(self._actions_taken)}"
        related = self.memory.retrieve_workflows(query, self._task_type, k=3)
        response = self._induce_or_integrate_workflow(related)
        data = parse_json_object(response)
        workflow = self._workflow_from_json(data)

        updated = None
        added = None
        target_id = self._target_workflow_id(data, related)
        if target_id is not None:
            existing = self.memory.get_workflow(target_id)
            if existing:
                updates = dict(workflow)
                updates["support"] = existing.get("support", 0) + 1
                updates["evidence_count"] = existing.get("evidence_count", 1) + 1
                updates["source_episodes"] = [self.episode]
                updated = self.memory.update_workflow(target_id, updates)

        if updated is None:
            workflow.update({
                "episode": self.episode,
                "task_type": self._task_type,
                "support": 1,
                "failure": 0,
                "evidence_count": 1,
                "source_episodes": [self.episode],
            })
            added = self.memory.add_workflow(workflow)

        self._episode_trace.append({
            "type": "memory_update",
            "kind": "workflow",
            "response": response,
            "related_workflow_ids": [item.get("id") for item in related],
            "added": added,
            "updated": updated,
            "workflow_feedback_updated": feedback_updated,
        })
        return {
            "workflow_feedback_updated": feedback_updated,
            "workflow_added": 1 if added else 0,
            "workflow_updated": 1 if updated else 0,
        }

    def _induce_or_integrate_workflow(self, related: list[dict[str, Any]]) -> str:
        related_text = self.memory.format_workflows(related) if self.memory and related else "(none)"
        prompt = [
            {"role": "system", "content": (
                "You implement Agent Workflow Memory for an embodied/text-game "
                "agent. From a successful trajectory, induce an abstract reusable "
                "workflow. If a related workflow already covers the same task "
                "pattern, integrate the new evidence by revising that workflow. "
                "Return strict JSON with keys: mode, target_id, title, task_pattern, "
                "steps, constraints. mode must be 'new' or 'update'. target_id is "
                "null for new workflows. steps and constraints must be arrays of "
                "short imperative natural-language strings."
            )},
            {"role": "user", "content": (
                f"Related workflows:\n{related_text}\n\n"
                f"Task type: {self._task_type}\n"
                f"Task: {self._task_obs}\n"
                f"Successful trajectory:\n{self._trajectory_text()}\n\n"
                "Induce or integrate one reusable workflow."
            )},
        ]
        return self.llm.chat(prompt, max_tokens=700 if self.llm.thinking_disabled else 1200)

    def _workflow_from_json(self, data: dict[str, Any]) -> dict[str, Any]:
        steps = data.get("steps")
        if not isinstance(steps, list) or not steps:
            steps = self._fallback_steps()
        constraints = data.get("constraints")
        if not isinstance(constraints, list):
            constraints = []
        title = data.get("title") or f"{self._task_type or 'task'} workflow"
        task_pattern = data.get("task_pattern") or self._task_obs
        return {
            "title": shorten(str(title), 120),
            "task_pattern": shorten(str(task_pattern), 260),
            "steps": [shorten(str(step), 220) for step in steps if str(step).strip()][:10],
            "constraints": [shorten(str(item), 220) for item in constraints if str(item).strip()][:6],
        }

    def _fallback_steps(self) -> list[str]:
        deduped = []
        seen = set()
        for action in self._actions_taken:
            key = action.lower().strip()
            if not key or key in seen:
                continue
            deduped.append(f"Use action '{action}' when the observation makes it valid.")
            seen.add(key)
            if len(deduped) >= 8:
                break
        return deduped or ["Inspect the current state, then choose a valid action that advances the task."]

    @staticmethod
    def _target_workflow_id(data: dict[str, Any], related: list[dict[str, Any]]) -> int | None:
        if data.get("mode") != "update":
            return None
        try:
            target_id = int(data.get("target_id"))
        except (TypeError, ValueError):
            return None
        related_ids = {item.get("id") for item in related}
        return target_id if target_id in related_ids else None
