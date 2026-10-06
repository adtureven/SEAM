"""RAP baseline."""

from __future__ import annotations

import json
import re
from typing import Any

from agent.baselines.memory import BaselineMemory
from agent.baselines.react import ReActAgent
from agent.baselines.utils import shorten
from env.adapter import EnvironmentAdapter
from llm.client import LLMClient


class RAPAgent(ReActAgent):
    """Budgeted RAP baseline: choose actions by LLM world-model planning."""

    method_name = "rap"

    def __init__(self, llm: LLMClient, adapter: EnvironmentAdapter,
                 memory: BaselineMemory | None = None, episode: int = 0,
                 max_history_steps: int = 6, max_candidates: int = 8,
                 rollout_depth: int = 2, branch_width: int = 3,
                 world_model_max_tokens: int = 512):
        super().__init__(
            llm=llm,
            adapter=adapter,
            memory=memory,
            episode=episode,
            max_history_steps=max_history_steps,
        )
        self.max_candidates = max_candidates
        self.rollout_depth = rollout_depth
        self.branch_width = branch_width
        self.world_model_max_tokens = world_model_max_tokens

    def _method_instruction(self) -> str:
        return (
            "Use reasoning via planning: simulate the likely effect of candidate "
            "actions, estimate which action best advances the task, then choose it."
        )

    def act(self, observation: str | None = None, candidates: list[str] | None = None,
            reward: float = 0.0) -> str:
        if observation is not None and len(self._history) > 2:
            self.observe(observation, reward=reward)

        if not candidates:
            return super().act(observation, candidates, reward)

        state = observation if observation is not None else self._task_obs
        call_idx_before = self.llm.total_calls
        action, plan_trace = self._plan_with_world_model(state, candidates)
        action = self._validate_action(action, candidates)
        self._actions_taken.append(action)
        self._history.append({"role": "user", "content": state})
        self._history.append({"role": "assistant", "content": f"Action: {action}"})
        self._trim_history()

        self._episode_trace.append({
            "type": "step",
            "step": len(self._actions_taken),
            "observation": observation,
            "action": action,
            "candidate_count": len(candidates),
            "planning": plan_trace,
            "llm_calls": [
                {
                    "call_idx_start": call_idx_before + 1,
                    "call_idx_end": self.llm.total_calls,
                    "num_planning_calls": self.llm.total_calls - call_idx_before,
                    "parsed_action": action,
                }
            ],
        })
        return action

    def _plan_with_world_model(self, state: str, candidates: list[str]) -> tuple[str, dict[str, Any]]:
        root_actions = self._root_candidates(candidates)
        branches = []
        for action in root_actions:
            score, trace = self._evaluate_branch(state, action, self.rollout_depth)
            branches.append({
                "action": action,
                "score": round(score, 4),
                "trace": trace,
            })
        branches.sort(key=lambda item: -item["score"])
        best = branches[0]["action"] if branches else (candidates[0] if candidates else "look")
        return best, {
            "algorithm": "rap_world_model_tree_search",
            "rollout_depth": self.rollout_depth,
            "branch_width": self.branch_width,
            "max_candidates": self.max_candidates,
            "world_model_max_tokens": self.world_model_max_tokens,
            "root_candidates": root_actions,
            "branches": branches,
        }

    def _root_candidates(self, candidates: list[str]) -> list[str]:
        deduped = []
        seen = set()
        for action in candidates:
            key = action.lower().strip()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(action)
        non_repeats = [a for a in deduped if a not in self._actions_taken[-2:]]
        return (non_repeats or deduped)[:self.max_candidates]

    def _evaluate_branch(self, state: str, action: str, depth: int) -> tuple[float, dict[str, Any]]:
        prediction = self._world_model_step(state, action)
        immediate = prediction["immediate_reward"]
        value = prediction["value"]
        trace = {
            "state": shorten(state, 300),
            "action": action,
            "predicted_next_state": prediction["next_state"],
            "immediate_reward": immediate,
            "value": value,
            "future": [],
        }
        if depth <= 1:
            return immediate + value, trace

        future_scores = []
        for future_action in prediction["next_actions"][:self.branch_width]:
            score, child = self._evaluate_branch(prediction["next_state"], future_action, depth - 1)
            future_scores.append(score)
            trace["future"].append(child)
        best_future = max(future_scores) if future_scores else value
        return immediate + 0.8 * best_future, trace

    def _world_model_step(self, state: str, action: str) -> dict[str, Any]:
        prompt = [
            {"role": "system", "content": (
                "You are the RAP world model and reward model for a text-game "
                "language agent. Predict the next state after an action and score "
                "task progress. Return strict JSON with keys: next_state, "
                "immediate_reward, value, next_actions. immediate_reward and value "
                "must be numbers from 0 to 1. next_actions must contain concise "
                "plausible future actions, not observations."
            )},
            {"role": "user", "content": (
                f"Task type: {self._task_type}\n"
                f"Task: {self._task_obs}\n"
                f"Recent real trajectory:\n{self._trajectory_text()}\n\n"
                f"Current or imagined state:\n{state}\n"
                f"Candidate action:\n{action}"
            )},
        ]
        response = self.llm.chat(
            prompt,
            max_tokens=self.world_model_max_tokens if self.llm.thinking_disabled
            else max(self.world_model_max_tokens, 1024),
        )
        return self._parse_world_model_response(response, state)

    def _parse_world_model_response(self, response: str, fallback_state: str) -> dict[str, Any]:
        text = (response or "").strip()
        data: dict[str, Any] = {}
        match = re.search(r"\{.*\}", text, flags=re.S)
        if match:
            try:
                data = json.loads(match.group(0))
            except json.JSONDecodeError:
                data = {}
        if not data:
            data = {
                "next_state": self._extract_field(text, "next_state") or fallback_state,
                "immediate_reward": self._extract_float(text, "immediate_reward"),
                "value": self._extract_float(text, "value"),
                "next_actions": self._extract_actions(text),
            }
        next_actions = data.get("next_actions") or []
        if isinstance(next_actions, str):
            next_actions = [a.strip() for a in re.split(r"[,;\n]", next_actions) if a.strip()]
        return {
            "next_state": shorten(str(data.get("next_state") or fallback_state), 500),
            "immediate_reward": self._clip01(data.get("immediate_reward", 0.0)),
            "value": self._clip01(data.get("value", 0.0)),
            "next_actions": [str(a).strip().lower() for a in next_actions if str(a).strip()][:self.branch_width],
        }

    @staticmethod
    def _clip01(value: Any) -> float:
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _extract_field(text: str, name: str) -> str:
        match = re.search(rf"(?im)^{name}\s*:\s*(.+)$", text)
        return match.group(1).strip() if match else ""

    @staticmethod
    def _extract_float(text: str, name: str) -> float:
        match = re.search(rf"(?im)^{name}\s*:\s*([0-9.]+)", text)
        if not match:
            return 0.0
        try:
            return float(match.group(1))
        except ValueError:
            return 0.0

    def _extract_actions(self, text: str) -> list[str]:
        actions = []
        valid = self.adapter.get_valid_commands()
        for line in text.splitlines():
            clean = re.sub(r"^[-*\d.)\s]+", "", line.strip()).lower()
            clean = re.sub(r"(?i)^next_actions?\s*:\s*", "", clean).strip()
            if any(clean.startswith(cmd) for cmd in valid):
                actions.append(clean)
        return actions
