"""ReAct baseline agent."""

from __future__ import annotations

import re
import time
from typing import Any

from agent.base_agent import BaseAgent
from agent.baselines.memory import BaselineMemory
from agent.baselines.utils import load_react_prompts, shorten
from env.adapter import EnvironmentAdapter
from llm.client import LLMClient


class ReActAgent(BaseAgent):
    """ReAct-style baseline: interleave brief reasoning with environment actions."""

    method_name = "react"

    def __init__(self, llm: LLMClient, adapter: EnvironmentAdapter,
                 memory: BaselineMemory | None = None, episode: int = 0,
                 max_history_steps: int = 8):
        super().__init__(llm, adapter=adapter)
        self.memory = memory
        self.episode = episode
        self.max_history_steps = max_history_steps
        self._task_obs = ""
        self._task_type = ""
        self._task_info: dict[str, Any] = {}
        self._trajectory: list[dict[str, Any]] = []
        self._last_observation = ""

    def reset(self, task_obs: str, task_type: str = ""):
        self._task_obs = task_obs
        self._task_info = self.adapter.parse_task(task_obs, task_type) if self.adapter else {}
        self._task_type = self._task_info.get("task_type") or task_type
        system_prompt = self._build_system_prompt()
        self._trajectory = []
        self._actions_taken = []
        self._episode_trace = []
        self._start_time = time.time()
        self._last_observation = task_obs

        memory_context = self._initial_memory_context()
        user_content = task_obs
        if memory_context:
            user_content += "\n\n" + memory_context

        self._history = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        self._episode_trace.append({
            "type": "init",
            "method": self.method_name,
            "task_type": self._task_type,
            "task_obs": task_obs,
            "system_prompt_len": len(system_prompt),
            "memory_context": memory_context or None,
        })

    def act(self, observation: str | None = None, candidates: list[str] | None = None,
            reward: float = 0.0) -> str:
        if observation is not None and len(self._history) > 2:
            self.observe(observation, reward=reward)
        user_msg = self._format_step_input(observation, candidates)
        if user_msg and (observation is not None or candidates):
            self._history.append({"role": "user", "content": user_msg})

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

    def observe(self, observation: str, reward: float = 0.0):
        if self._actions_taken:
            entry = {
                "action": self._actions_taken[-1],
                "observation": observation,
                "reward": reward,
            }
            if not self._trajectory or self._trajectory[-1] != entry:
                self._trajectory.append(entry)
        self._last_observation = observation

    def on_episode_end(self, success: bool):
        experience = self._store_experience(success)
        try:
            updates = self._update_memory_on_episode_end(success, experience)
        except Exception as exc:
            updates = {"error": str(exc)}
        self._episode_trace.append({
            "type": "result",
            "success": success,
            "total_steps": len(self._actions_taken),
            "total_llm_calls": self.llm.total_calls,
            "duration_seconds": round(time.time() - self._start_time, 2),
            "memory_updates": updates,
        })

    def _build_system_prompt(self) -> str:
        base = self.adapter.get_system_prompt() if self.adapter else "You are an agent."
        action_text = self.adapter.get_action_list_text() if self.adapter else ""
        few_shot = self._get_few_shot_context()
        if few_shot:
            base += "\n\nExamples:\n" + few_shot
        return (
            f"{base}\n\n{action_text}\n\n"
            f"{self._method_instruction()}\n\n"
            "Rules:\n"
            "- You may write one short Thought line.\n"
            "- The final line must be exactly: Action: <one valid action> or > <one valid action>.\n"
            "- Do not invent observations or simulate environment responses."
        )

    def _method_instruction(self) -> str:
        return (
            "Use the ReAct pattern: reason briefly about the current observation, "
            "then choose the next environment action."
        )

    def _initial_memory_context(self) -> str:
        return ""

    def _update_memory_on_episode_end(self, success: bool,
                                      experience: dict[str, Any]) -> dict[str, Any]:
        return {}

    def _get_few_shot_context(self, num_examples: int = 2) -> str:
        prompts = load_react_prompts()
        if not prompts or not self.adapter:
            return ""
        key = self.adapter.get_few_shot_key(self._task_type)
        examples = []
        for i in range(num_examples):
            item = prompts.get(f"react_{key}_{i}")
            if item:
                examples.append(item)
        return "\n\n".join(examples)

    def _trajectory_text(self) -> str:
        if self.adapter:
            return self.adapter.format_trajectory_for_reflection(self._trajectory)
        return "\n".join(
            f"- {entry.get('action', '')} -> {shorten(entry.get('observation', ''), 180)}"
            for entry in self._trajectory[-15:]
        )

    def _store_experience(self, success: bool) -> dict[str, Any]:
        record = {
            "episode": self.episode,
            "task_type": self._task_type,
            "task_obs": self._task_obs,
            "success": success,
            "actions": list(self._actions_taken),
            "trajectory": list(self._trajectory[-20:]),
        }
        if self.memory:
            return self.memory.add_experience(record)
        return record

    def _action_tokens(self) -> int:
        return 256 if self.llm.thinking_disabled else 1024

    def _trim_history(self):
        keep = 2 + 2 * self.max_history_steps
        if len(self._history) > keep:
            self._history = self._history[:2] + self._history[-(keep - 2):]

    def _parse_action(self, response: str) -> str:
        valid_cmds = self.adapter.get_valid_commands() if self.adapter else (
            "look", "inventory", "go to", "go", "open", "close", "take", "put",
            "move", "heat", "clean", "cool", "use", "examine",
        )
        lines = [line.strip() for line in (response or "").splitlines() if line.strip()]
        for line in reversed(lines):
            line = line.strip()
            if line.startswith(">"):
                line = line[1:].strip()
            line = re.sub(r"(?i)^(final\s+)?action\s*:\s*", "", line).strip()
            line = re.sub(r"^\d+[\.)]\s*", "", line).strip()
            line_low = line.lower()
            if line_low.startswith(("thought:", "think:", "plan:", "observation:", "reason:")):
                continue
            if any(line_low.startswith(cmd) for cmd in valid_cmds):
                return line_low
        return "look"
