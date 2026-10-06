"""
ExperimentLogger: standardized experiment output management.

Creates a structured output directory per experiment run with:
- config snapshot, metadata, summary
- per-episode JSONL traces (full LLM I/O)
- memory graph and evolution snapshots
- prompt recordings
"""

import json
import yaml
import time
from datetime import datetime
from pathlib import Path


class ExperimentLogger:
    def __init__(self, method: str, model_short: str, config: dict,
                 base_dir: str = "results"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path(base_dir) / f"{method}_{model_short}_{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "episodes").mkdir(exist_ok=True)
        (self.run_dir / "memory").mkdir(exist_ok=True)
        (self.run_dir / "prompts").mkdir(exist_ok=True)

        self._config = config
        self._method = method
        self._model_short = model_short
        self._start_time = time.time()
        self._start_iso = datetime.now().isoformat(timespec="seconds")

        with open(self.run_dir / "config.yaml", "w") as f:
            yaml.dump(self._redact_secrets(config), f, default_flow_style=False, allow_unicode=True)

    @property
    def llm_log_path(self) -> Path:
        return self.run_dir / "llm_calls.jsonl"

    def _redact_secrets(self, value):
        if isinstance(value, dict):
            redacted = {}
            for key, item in value.items():
                if key.lower() in {
                    "api_key",
                    "api_key_file",
                    "token",
                    "secret",
                    "password",
                    "base_url",
                } and item:
                    redacted[key] = "<redacted>"
                else:
                    redacted[key] = self._redact_secrets(item)
            return redacted
        if isinstance(value, list):
            return [self._redact_secrets(item) for item in value]
        return value

    def episode_path(self, idx: int) -> Path:
        return self.run_dir / "episodes" / f"ep_{idx:04d}.jsonl"

    def save_episode(self, idx: int, trace: list[dict]):
        path = self.episode_path(idx)
        with open(path, "w") as f:
            for entry in trace:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def save_prompts(self, system_prompt: str, few_shot_examples: dict | None = None):
        with open(self.run_dir / "prompts" / "system_prompt.txt", "w") as f:
            f.write(system_prompt)
        if few_shot_examples:
            with open(self.run_dir / "prompts" / "few_shot_examples.json", "w") as f:
                json.dump(few_shot_examples, f, indent=2, ensure_ascii=False)

    def save_memory(self, graph, evolution):
        with open(self.run_dir / "memory" / "graph_final.json", "w") as f:
            data = graph.to_dict() if hasattr(graph, "to_dict") else graph.get_stats()
            json.dump(data, f, indent=2, ensure_ascii=False)
        if hasattr(evolution, "_evolution_log"):
            with open(self.run_dir / "memory" / "evolution_log.json", "w") as f:
                json.dump(evolution._evolution_log, f, indent=2, ensure_ascii=False)

    def finalize(self, summary: dict):
        end_time = time.time()
        metadata = {
            "experiment_type": self._method,
            "model": self._config["llm"]["model"],
            "model_short": self._model_short,
            "dataset": self._config.get("_env_name", "alfworld"),
            "split": self._config["env"].get("split", ""),
            "num_episodes": summary.get("num_episodes"),
            "max_steps": summary.get("max_steps", self._config["env"].get("max_steps")),
            "start_time": self._start_iso,
            "end_time": datetime.now().isoformat(timespec="seconds"),
            "duration_seconds": round(end_time - self._start_time, 1),
            "prompt_style": "react_few_shot",
            "memory_config": self._config.get("memory", {}),
            "llm_config": self._redact_secrets({
                k: v for k, v in self._config["llm"].items()
                if k not in ("api_key",)
            }),
        }
        with open(self.run_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        with open(self.run_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"Results saved to: {self.run_dir}")
