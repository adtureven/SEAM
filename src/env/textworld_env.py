"""TextWorld environment wrapper."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class TextWorldTaskInfo:
    task_name: str
    task_desc: str
    task_type: str = "textworld"
    difficulty: str = "unknown"
    game_file: str = ""
    max_score: float = 0.0


class TextWorldEnv:
    """Adapter around TextWorld's Gym interface.

    Games are generated ahead of time as .z8 files. This keeps all methods on
    the same fixed benchmark suite and avoids run-time randomness.
    """

    def __init__(self, games_dir: str = "artifacts/textworld/games",
                 difficulties: list[str] | None = None, num_tasks: int | None = None,
                 home_dir: str = "artifacts/textworld/.home", max_steps: int = 50,
                 order: str = "interleaved", suite_name: str = ""):
        self.games_dir = Path(games_dir).expanduser()
        self.difficulties = difficulties or ["easy", "medium", "hard"]
        self.configured_num_tasks = num_tasks
        self.home_dir = Path(home_dir).expanduser()
        self.max_steps = int(max_steps)
        self.order = order
        self.suite_name = suite_name.strip().lower()
        self._step_count = 0
        self._env = None
        self._last_infos: dict[str, Any] = {}
        self._current_task = TextWorldTaskInfo("textworld#0", "")
        self._treasure_target_cache: dict[str, str] = {}

        self.home_dir.mkdir(parents=True, exist_ok=True)
        os.environ["HOME"] = str(self.home_dir.resolve())
        self._games = self._discover_games()

    def _discover_games(self) -> list[tuple[Path, str]]:
        if not self.games_dir.exists():
            raise FileNotFoundError(
                f"TextWorld games directory is missing: {self.games_dir}. "
                "Run `bash scripts/generate_textworld_games.sh` first."
            )

        allowed = set(self.difficulties)
        by_difficulty: dict[str, list[tuple[Path, str]]] = {}
        for difficulty in self.difficulties:
            level_dir = self.games_dir / difficulty
            if not level_dir.exists():
                continue
            by_difficulty[difficulty] = []
            for path in sorted(level_dir.glob("*.z8")):
                by_difficulty[difficulty].append((path.resolve(), difficulty))

        games: list[tuple[Path, str]] = []
        if self.order == "interleaved" and by_difficulty:
            max_len = max(len(items) for items in by_difficulty.values())
            for idx in range(max_len):
                for difficulty in self.difficulties:
                    items = by_difficulty.get(difficulty, [])
                    if idx < len(items):
                        games.append(items[idx])
        else:
            for difficulty in self.difficulties:
                games.extend(by_difficulty.get(difficulty, []))

        if not games:
            for path in sorted(self.games_dir.glob("*.z8")):
                difficulty = self._infer_difficulty(path)
                if difficulty in allowed or not allowed:
                    games.append((path.resolve(), difficulty))

        if not games:
            raise FileNotFoundError(
                f"No .z8 TextWorld games found under {self.games_dir}. "
                "Run `bash scripts/generate_textworld_games.sh` first."
            )

        if self.configured_num_tasks is not None:
            games = games[: int(self.configured_num_tasks)]
        return games

    @staticmethod
    def _infer_difficulty(path: Path) -> str:
        text = f"{path.parent.name}_{path.name}".lower()
        for difficulty in ("easy", "medium", "hard"):
            if difficulty in text:
                return difficulty
        return "unknown"

    def num_tasks(self) -> int:
        return len(self._games)

    def reset(self, task_idx: int) -> tuple[str, TextWorldTaskInfo]:
        self.close()
        self._step_count = 0
        game_file, difficulty = self._games[task_idx % len(self._games)]
        self._env = self._make_env(game_file)
        obs, infos = self._env.reset()
        raw_infos = dict(infos or {})
        self._last_infos = self._prepare_infos(dict(raw_infos), game_file)
        obs = self._prepare_observation(obs, raw_infos, self._last_infos)
        task_desc = self._task_desc_from_infos(obs, self._last_infos)
        max_score = float(self._last_infos.get("max_score") or 0.0)
        task_type = f"textworld_{difficulty}"
        if self.suite_name:
            task_type = f"textworld_{self.suite_name}_{difficulty}"
        self._current_task = TextWorldTaskInfo(
            task_name=game_file.stem,
            task_desc=task_desc,
            task_type=task_type,
            difficulty=difficulty,
            game_file=str(game_file),
            max_score=max_score,
        )
        return self._format_observation(obs, self._last_infos), self._current_task

    def _make_env(self, game_file: Path):
        try:
            import textworld
            import textworld.gym
        except Exception as exc:
            raise ImportError(
                "TextWorld support requires the `textworld` package in the active environment. "
                "Install with `python -m pip install textworld`."
            ) from exc

        request_infos = textworld.EnvInfos(
            admissible_commands=True,
            description=True,
            inventory=True,
            objective=True,
            score=True,
            max_score=True,
            won=True,
            lost=True,
        )
        env_id = textworld.gym.register_game(
            str(game_file),
            request_infos=request_infos,
            max_episode_steps=self.max_steps,
        )
        return textworld.gym.make(env_id)

    def _prepare_observation(
        self,
        obs: str,
        raw_infos: dict[str, Any],
        prepared_infos: dict[str, Any],
    ) -> str:
        if self.suite_name != "treasure":
            return obs
        original_objective = str(raw_infos.get("objective") or "").strip()
        masked_objective = str(prepared_infos.get("objective") or "").strip()
        if original_objective and masked_objective and original_objective in obs:
            return obs.replace(original_objective, masked_objective)
        return obs

    def _prepare_infos(self, infos: dict[str, Any], game_file: Path) -> dict[str, Any]:
        if self.suite_name != "treasure":
            return infos

        cache_key = str(game_file)
        if cache_key not in self._treasure_target_cache:
            self._treasure_target_cache[cache_key] = self._treasure_target_from_metadata(game_file)
        target = self._treasure_target_cache[cache_key]
        if not target:
            target = self._treasure_target_from_objective(str(infos.get("objective") or ""))
        if target:
            infos["objective"] = (
                f"Explore the map, find the {target}, and pick it up to finish the game."
            )
        else:
            infos["objective"] = "Explore the map, find the target item, and pick it up to finish the game."
        return infos

    @staticmethod
    def _treasure_target_from_metadata(game_file: Path) -> str:
        metadata_file = game_file.with_suffix(".json")
        if not metadata_file.exists():
            return ""
        try:
            data = json.loads(metadata_file.read_text())
        except Exception:
            return ""
        walkthrough = ((data.get("metadata") or {}).get("walkthrough") or [])
        for action in reversed(walkthrough):
            target = TextWorldEnv._target_from_take_action(str(action))
            if target:
                return target
        return ""

    @staticmethod
    def _target_from_take_action(action: str) -> str:
        match = re.match(r"\s*take\s+(.+?)(?:\s+from\s+.+)?$", action, flags=re.I)
        if not match:
            return ""
        return re.sub(r"\s+", " ", match.group(1)).strip(" .,:;").lower()

    @staticmethod
    def _treasure_target_from_objective(objective: str) -> str:
        patterns = (
            r"\b(?:pick[- ]?up|retrieve|recover|lift)\s+(?:the\s+|a\s+|an\s+)?(.+?)(?:\s+from\b|[.!?;,]|$)",
            r"\btake\s+(?!a\s+trip\b|a\s+walk\b|a\s+look\b)(?:the\s+|a\s+|an\s+)?(.+?)(?:\s+from\b|[.!?;,]|$)",
            r"\bget your hands on\s+(?:the\s+|a\s+|an\s+)?(.+?)(?:\s+from\b|[.!?;,]|$)",
        )
        matches: list[str] = []
        for pattern in patterns:
            matches.extend(re.findall(pattern, objective, flags=re.I))
        if not matches:
            return ""
        target = matches[-1]
        return re.sub(r"\s+", " ", target).strip(" .,:;").lower()

    def step(self, action: str) -> tuple[str, float, bool, dict]:
        if self._env is None:
            raise RuntimeError("TextWorldEnv.step() called before reset().")
        self._step_count += 1
        obs, reward, done, infos = self._env.step(action)
        raw_infos = dict(infos or {})
        infos = dict(raw_infos)
        if self._current_task.game_file:
            infos = self._prepare_infos(infos, Path(self._current_task.game_file))
            obs = self._prepare_observation(obs, raw_infos, infos)
        score = float(infos.get("score") if infos.get("score") is not None else reward or 0.0)
        max_score = float(infos.get("max_score") or self._current_task.max_score or 0.0)
        normalized_score = score / max_score if max_score > 0 else 0.0
        if self._step_count >= self.max_steps:
            done = True
            obs = f"{obs}\nYou have run out of steps."
        infos["steps"] = self._step_count
        infos["score"] = score
        infos["max_score"] = max_score
        infos["normalized_score"] = normalized_score
        infos["goal_reached"] = bool(infos.get("won") or (done and max_score > 0 and score >= max_score))
        self._last_infos = infos
        return self._format_observation(obs, infos), score, bool(done), infos

    def get_available_actions(self) -> list[str]:
        commands = self._last_infos.get("admissible_commands") or []
        if isinstance(commands, (list, tuple)):
            deduped = []
            seen = set()
            for command in commands:
                text = str(command).strip()
                if text and text not in seen:
                    deduped.append(text)
                    seen.add(text)
            if deduped:
                return deduped
        return ["look", "inventory"]

    def close(self):
        if self._env is not None:
            try:
                self._env.close()
            except Exception:
                pass
            self._env = None

    @staticmethod
    def _task_desc_from_infos(obs: str, infos: dict[str, Any]) -> str:
        objective = str(infos.get("objective") or "").strip()
        if objective:
            return objective
        for marker in ("Objective:", "Goal:", "Your objective is"):
            idx = obs.lower().find(marker.lower())
            if idx >= 0:
                return obs[idx + len(marker):].strip().splitlines()[0]
        return ""

    @staticmethod
    def _format_observation(obs: str, infos: dict[str, Any]) -> str:
        parts = []
        objective = str(infos.get("objective") or "").strip()
        if objective and objective.lower() not in obs.lower():
            parts.append(f"Objective: {objective}")
        inventory = str(infos.get("inventory") or "").strip()
        if inventory and inventory.lower() not in obs.lower():
            parts.append(f"Inventory: {inventory}")
        score = infos.get("score")
        max_score = infos.get("max_score")
        if score is not None and max_score is not None:
            parts.append(f"Score: {score}/{max_score}")
        parts.append(str(obs))
        return "\n\n".join(part for part in parts if part)
