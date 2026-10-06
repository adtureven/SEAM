"""
Optional ScienceWorld environment wrapper.

This wrapper keeps ScienceWorld support behind an optional dependency. It exposes
the same small interface used by experiments: num_tasks, reset, step, and
get_available_actions.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SciWorldTaskInfo:
    task_name: str
    task_desc: str
    task_type: str
    variation_idx: int = 0
    game_file: str = ""


class SciWorldEnv:
    def __init__(self, task_names: list[str] | None = None,
                 split: str = "test",
                 variation_idx: int = 0,
                 simplification: str = "easy",
                 jar_path: str | None = None,
                 expand_variations: bool = False,
                 task_indices: list[int] | None = None,
                 max_steps: int = 50):
        self.max_steps = max_steps
        self.split = split
        self.variation_idx = variation_idx
        self.simplification = simplification
        self.jar_path = jar_path
        self.expand_variations = expand_variations
        self.task_indices = [int(idx) for idx in (task_indices or [])]
        self.task_names = task_names or []
        self._tasks: list[tuple[str, int]] = []
        self._task_original_indices: list[int] = []
        self._step_count = 0
        self._done = False
        self._last_non_negative_score = 0.0
        self._env = self._make_env()
        if not self.task_names:
            self.task_names = self._discover_task_names()
        self._tasks, self._task_original_indices = self._select_task_instances(
            self._build_task_instances()
        )
        self._last_info = {}

    def _make_env(self):
        try:
            from scienceworld import ScienceWorldEnv as _ScienceWorldEnv
            kwargs = {"envStepLimit": self.max_steps}
            if self.jar_path:
                kwargs["jarPath"] = self.jar_path
            try:
                return _ScienceWorldEnv("", **kwargs)
            except TypeError:
                return _ScienceWorldEnv("", envStepLimit=self.max_steps)
        except Exception:
            try:
                from sciworld import SciWorldEnv as _SciWorldEnv
                return _SciWorldEnv()
            except Exception as exc:
                raise ImportError(
                    "ScienceWorld support requires an installed `scienceworld` package. "
                    "Install the benchmark package, then rerun with --env scienceworld."
                ) from exc

    def _discover_task_names(self) -> list[str]:
        for attr in ("getTaskNames", "get_task_names"):
            if hasattr(self._env, attr):
                try:
                    names = getattr(self._env, attr)()
                    if names:
                        return [str(name) for name in names]
                except Exception:
                    pass
        return ["scienceworld_task"]

    def num_tasks(self) -> int:
        return len(self._tasks)

    def reset(self, task_idx: int) -> tuple[str, SciWorldTaskInfo]:
        self._step_count = 0
        self._done = False
        self._last_non_negative_score = 0.0
        task_slot = task_idx % len(self._tasks)
        task_name, variation_idx = self._tasks[task_slot]
        original_idx = self._task_original_indices[task_slot]

        obs = ""
        if hasattr(self._env, "load"):
            try:
                self._env.load(task_name, variation_idx, self.simplification)
            except TypeError:
                try:
                    self._env.load(task_name, variation_idx)
                except TypeError:
                    self._env.load(task_name)
        reset_result = self._env.reset()
        if isinstance(reset_result, tuple):
            obs = str(reset_result[0])
            if len(reset_result) > 1 and isinstance(reset_result[1], dict):
                self._last_info = reset_result[1]
        else:
            obs = str(reset_result)
        if hasattr(self._env, "getTaskDescription"):
            task_desc = str(self._env.getTaskDescription())
        else:
            task_desc = task_name
        if "task:" not in obs.lower() and "your task is to:" not in obs.lower():
            obs = f"{obs}\n\nYour task is to: {task_desc}"
        task_id = f"{task_name}#v{variation_idx}" if self.expand_variations else task_name
        if self.task_indices:
            task_id = f"{task_id}@idx{original_idx}"
        return obs, SciWorldTaskInfo(
            task_name=task_id,
            task_desc=task_desc,
            task_type=task_name,
            variation_idx=variation_idx,
        )

    def _build_task_instances(self) -> list[tuple[str, int]]:
        if not self.expand_variations:
            return [(name, self.variation_idx) for name in self.task_names]

        instances: list[tuple[str, int]] = []
        for name in self.task_names:
            max_variations = self._get_max_variations(name)
            instances.extend((name, idx) for idx in range(max_variations))
        return instances

    def _select_task_instances(self, all_tasks: list[tuple[str, int]]) -> tuple[list[tuple[str, int]], list[int]]:
        if not self.task_indices:
            return all_tasks, list(range(len(all_tasks)))
        selected: list[tuple[str, int]] = []
        original_indices: list[int] = []
        for idx in self.task_indices:
            if idx < 0 or idx >= len(all_tasks):
                raise IndexError(
                    f"ScienceWorld task index {idx} is outside available range 0..{len(all_tasks) - 1}"
                )
            selected.append(all_tasks[idx])
            original_indices.append(idx)
        if not selected:
            raise ValueError("scienceworld.task_indices was provided but no tasks were selected.")
        return selected, original_indices

    def _get_max_variations(self, task_name: str) -> int:
        for attr in ("getMaxVariations", "get_max_variations"):
            if hasattr(self._env, attr):
                try:
                    value = getattr(self._env, attr)(task_name)
                    if int(value) > 0:
                        return int(value)
                except Exception:
                    pass
        return 1

    def step(self, action: str) -> tuple[str, float, bool, dict]:
        if self._done:
            return "Game is over.", 0.0, True, {}
        self._step_count += 1
        result = self._env.step(action)
        obs, reward, done, info = self._normalize_step_result(result)
        if self._step_count >= self.max_steps:
            done = True
            obs += "\nYou have run out of steps."
        self._done = done
        info = dict(info)
        score = self._safe_float(info.get("score"))
        if score is not None and score >= 0:
            self._last_non_negative_score = score
        info["non_negative_score"] = self._last_non_negative_score
        info["score_for_eval"] = (
            self._last_non_negative_score
            if score is not None and score < 0
            else score
        )
        info["steps"] = self._step_count
        info.setdefault("goal_reached", bool(done and reward > 0))
        return obs, reward, done, info

    @staticmethod
    def _safe_float(value):
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    def get_available_actions(self) -> list[str]:
        for attr in ("getValidActionObjectCombinations", "getPossibleActions",
                     "get_possible_actions", "getValidActions"):
            if hasattr(self._env, attr):
                try:
                    actions = getattr(self._env, attr)()
                    if actions:
                        return [str(a) for a in actions]
                except Exception:
                    pass
        if self._last_info.get("admissible_commands"):
            return [str(a) for a in self._last_info["admissible_commands"]]
        return ["look", "inventory"]

    def _normalize_step_result(self, result) -> tuple[str, float, bool, dict]:
        if isinstance(result, tuple):
            if len(result) == 4:
                obs, reward, done, info = result
                self._last_info = info if isinstance(info, dict) else {}
                return str(obs), float(reward or 0.0), bool(done), self._last_info
            if len(result) == 3:
                obs, reward, done = result
                return str(obs), float(reward or 0.0), bool(done), {}
        return str(result), 0.0, False, {}

    def close(self):
        if hasattr(self._env, "close"):
            try:
                self._env.close()
            except Exception:
                pass
