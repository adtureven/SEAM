"""Wrapper for the text-only Princeton WebShop Gym environment."""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class WebShopTaskInfo:
    task_name: str
    task_desc: str
    task_type: str = "webshop"
    difficulty: str = "unknown"
    game_file: str = ""
    max_score: float = 1.0


class WebShopEnv:
    """Normalize WebShop's legacy Gym API to the SEAM environment interface.

    ``webshop_repo`` should point to a Princeton WebShop checkout when the
    package is not already importable. Episodes are selected by the integer
    session index so sequential runs use a fixed instruction ordering.
    """

    def __init__(self, webshop_repo: str | None = None,
                 observation_mode: str = "text", num_products: int | None = None,
                 limit_goals: int = -1, max_steps: int = 50,
                 human_goals: bool = True, show_attrs: bool = False,
                 split: str = "all"):
        if split not in {"all", "test", "eval", "train"}:
            raise ValueError(f"Unknown WebShop split: {split}")
        if split != "all" and (num_products is not None or limit_goals != -1 or not human_goals):
            raise ValueError("Official WebShop splits require all products, limit_goals=-1, and human_goals=true.")
        if webshop_repo:
            repo = str(Path(webshop_repo).expanduser().resolve())
            if repo not in sys.path:
                sys.path.insert(0, repo)

        try:
            from web_agent_site.envs import WebAgentTextEnv
        except ImportError as exc:
            raise ImportError(
                "WebShop is not installed. Clone Princeton-NLP/WebShop, run its "
                "setup script, and set webshop.repo_path in the experiment config."
            ) from exc

        self.max_steps = int(max_steps)
        self._step_count = 0
        self._current_task = WebShopTaskInfo("webshop#0", "")
        self._env = WebAgentTextEnv(
            observation_mode=observation_mode,
            num_products=num_products,
            limit_goals=limit_goals,
            human_goals=int(human_goals),
            show_attrs=show_attrs,
        )
        goals = getattr(getattr(self._env, "server", None), "goals", None)
        if goals is None:
            goals = getattr(getattr(self._env, "unwrapped", None), "server", None)
            goals = getattr(goals, "goals", None)
        self._num_tasks = len(goals) if goals is not None else max(1, limit_goals)
        # Official upstream split uses the goals shuffled by SimServer (seed 233).
        if split != "all" and (goals is None or len(goals) < 1500):
            raise ValueError("Official WebShop splits require the complete goal list (at least 1500 goals).")
        if split == "test":
            self._task_indices = list(range(500))
        elif split == "eval":
            self._task_indices = list(range(500, 1500))
        elif split == "train":
            self._task_indices = list(range(1500, self._num_tasks))
        else:
            self._task_indices = list(range(self._num_tasks))
        self._num_tasks = len(self._task_indices)
        self._last_obs = ""
        self._last_reward = 0.0

    def num_tasks(self) -> int:
        return self._num_tasks

    def reset(self, task_idx: int) -> tuple[str, WebShopTaskInfo]:
        self._step_count = 0
        self._last_reward = 0.0
        if not 0 <= task_idx < self._num_tasks:
            raise IndexError(f"WebShop task index out of range: {task_idx}")
        session_idx = self._task_indices[task_idx]
        result = self._env.reset(session=session_idx)
        # WebShop returns (observation, None); tolerate Gym wrappers that return
        # an observation directly or the newer (observation, info) convention.
        raw_obs = result[0] if isinstance(result, tuple) else result
        instruction = self._get_instruction(raw_obs)
        self._current_task = WebShopTaskInfo(
            task_name=f"webshop#{session_idx}",
            task_desc=instruction,
        )
        self._last_obs = self._format_observation(raw_obs)
        return self._last_obs, self._current_task

    def step(self, action: str) -> tuple[str, float, bool, dict[str, Any]]:
        self._step_count += 1
        result = self._env.step(action)
        if len(result) == 5:
            raw_obs, reward, terminated, truncated, info = result
            done = bool(terminated or truncated)
        else:
            raw_obs, reward, done, info = result
            info = dict(info or {})
        reward = float(reward or 0.0)
        self._last_reward = reward
        success = bool(done and reward >= 0.99)
        if self._step_count >= self.max_steps and not done:
            done = True
        info.update({
            "goal_reached": success,
            "score": reward,
            "score_for_eval": reward,
            "max_score": 1.0,
            "normalized_score": reward,
            "taskName": self._current_task.task_name,
        })
        self._last_obs = self._format_observation(raw_obs, reward=reward, done=done)
        return self._last_obs, reward, bool(done), info

    def get_available_actions(self) -> list[str]:
        state = self._env.get_available_actions()
        actions = [f"click[{text}]" for text in state.get("clickables", [])]
        if state.get("has_search_bar"):
            # This is a prompt template. WebShopAdapter permits the model to
            # replace its contents with a query not known in advance.
            actions.insert(0, "search[query]")
        return actions

    def close(self):
        self._env.close()

    def _get_instruction(self, observation: str) -> str:
        try:
            return str(self._env.get_instruction_text()).strip()
        except Exception:
            match = re.search(r"Instruction:\s*(.+)", str(observation), flags=re.I)
            return match.group(1).strip() if match else str(observation).strip()

    def _format_observation(self, observation: str, reward: float = 0.0,
                            done: bool = False) -> str:
        url = str(getattr(getattr(self._env, "browser", None), "current_url", ""))
        page = "home"
        for name in ("search_results", "item_page", "item_sub_page", "done"):
            if name in url:
                page = name
                break
        return (
            f"WebShop page: {page}\n"
            f"Instruction: {self._current_task.task_desc}\n"
            f"Reward: {float(reward):.4f}; Done: {str(bool(done)).lower()}\n"
            f"{str(observation)}"
        )
