"""
ALFWorld ablation experiments for Seam modules.

Variants:
- no_self_evolution
- no_evidence_verification
- no_causal_abstraction
- no_memory_revision
- no_effect_prediction
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agent.secm_agent import SEAMAgent, _load_react_prompts
from env.alfworld_adapter import ALFWorldAdapter
from env.alfworld_pddl import ALFWorldEnv
from llm.client import make_llm_from_config
from memory.causal_graph import HierarchicalCausalGraph
from memory.evolution import EvolutionEngine
from utils.logger import ExperimentLogger


MODEL_SHORT_NAMES = {
    "mimo-v2.5": "mimo25",
    "qwen3.5-35b-a3b": "qwen35",
    "qwen3-4b": "qwen3-4b",
    "qwen3.5-4b": "qwen35_4b",
    "qwen3.5-9b": "qwen35_9b",
}


ABLATIONS = {
    "no_self_evolution": {
        "display": "No self-evolution",
        "engine": {"enable_self_evolution": False},
        "agent": {},
    },
    "no_evidence_verification": {
        "display": "No evidence-based verification",
        "engine": {"enable_verification": False},
        "agent": {},
    },
    "no_causal_abstraction": {
        "display": "No causal abstraction",
        "engine": {"enable_abstraction": False},
        "agent": {},
    },
    "no_memory_revision": {
        "display": "No memory revision",
        "engine": {"enable_revision": False, "enable_pruning": False},
        "agent": {},
    },
    "no_effect_prediction": {
        "display": "No effect prediction",
        "engine": {},
        "agent": {"enable_effect_prediction": False},
    },
}

ALIASES = {
    "self-evolution": "no_self_evolution",
    "evidence-based-verification": "no_evidence_verification",
    "evidence_verification": "no_evidence_verification",
    "causal-abstraction": "no_causal_abstraction",
    "causal_abstraction": "no_causal_abstraction",
    "memory-revision": "no_memory_revision",
    "memory_revision": "no_memory_revision",
    "effect-prediction": "no_effect_prediction",
    "effect_prediction": "no_effect_prediction",
}


def canonical_variant(variant: str) -> str:
    key = variant.strip().lower()
    key = ALIASES.get(key, key)
    if key not in ABLATIONS:
        allowed = ", ".join(list(ABLATIONS) + ["all"])
        raise ValueError(f"Unknown ablation variant: {variant}. Expected one of: {allowed}")
    return key


def make_evolution(config: dict, graph: HierarchicalCausalGraph,
                   adapter: ALFWorldAdapter, variant: str) -> EvolutionEngine:
    mem_cfg = config["memory"]
    engine_flags = {
        "enable_self_evolution": True,
        "enable_extraction": True,
        "enable_verification": True,
        "enable_abstraction": True,
        "enable_revision": True,
        "enable_pruning": True,
    }
    engine_flags.update(ABLATIONS[variant]["engine"])
    return EvolutionEngine(
        graph=graph,
        adapter=adapter,
        aggregation_threshold=mem_cfg.get("aggregation_threshold", 2),
        abstraction_min_members=mem_cfg.get("abstraction_min_members", 2),
        abstraction_min_confidence=mem_cfg.get("abstraction_min_confidence", 0.6),
        revision_error_threshold=mem_cfg.get("revision_error_threshold", 0.5),
        **engine_flags,
    )


def run_ablation(variant: str, config_path: str = "configs/seam_aaai2026.yaml",
                 num_episodes: int | None = None, max_steps: int | None = None) -> dict:
    variant = canonical_variant(variant)
    with open(config_path) as f:
        config = yaml.safe_load(f)
    config["_env_name"] = "alfworld"
    config["ablation"] = {
        "variant": variant,
        "display": ABLATIONS[variant]["display"],
        "engine_flags": ABLATIONS[variant]["engine"],
        "agent_flags": ABLATIONS[variant]["agent"],
    }

    steps_limit = int(max_steps or config["env"].get("max_steps", 50))
    model = config["llm"]["model"]
    model_short = MODEL_SHORT_NAMES.get(model, model.split("/")[-1][:10])
    logger = ExperimentLogger(
        method=f"seam_ablate_{variant}_alfworld",
        model_short=model_short,
        config=config,
        base_dir=config["experiment"]["log_dir"],
    )
    react_prompts = _load_react_prompts()
    logger.save_prompts(
        system_prompt="[ALFWorld system prompt with Seam ablation flags]",
        few_shot_examples=react_prompts if react_prompts else {},
    )

    print(f"=== Seam Ablation: {variant} [{ABLATIONS[variant]['display']}] ===")
    print(f"Output: {logger.run_dir}")

    env = ALFWorldEnv(
        data_path=config["env"]["data_path"],
        split=config["env"]["split"],
        max_steps=steps_limit,
    )
    adapter = ALFWorldAdapter()
    mem_cfg = config["memory"]
    graph = HierarchicalCausalGraph(
        stability_threshold=mem_cfg["stability_threshold"],
        stable_reliability=mem_cfg.get("stable_reliability", 0.7),
        revision_error_threshold=mem_cfg.get("revision_error_threshold", 0.5),
        aggregation_threshold=mem_cfg.get("aggregation_threshold", 2),
        adapter=adapter,
    )
    evolution = make_evolution(config, graph, adapter, variant)
    agent_flags = dict(ABLATIONS[variant]["agent"])

    n_episodes = min(int(num_episodes or config["experiment"]["num_episodes"]), env.num_tasks())
    results = []
    success_count = 0
    total_calls = 0
    type_stats = defaultdict(lambda: {"success": 0, "total": 0, "steps": []})
    start_time = time.time()

    for i in range(n_episodes):
        llm = make_llm_from_config(config, log_path=logger.llm_log_path)
        agent_cfg = {"enable_effect_prediction": mem_cfg.get("enable_effect_prediction", True)}
        agent_cfg.update(agent_flags)
        agent = SEAMAgent(
            llm,
            graph,
            evolution,
            adapter=adapter,
            episode=i,
            enable_llm_reflection=mem_cfg.get("enable_llm_reflection", False),
            retrieval_top_k=mem_cfg.get("retrieval_top_k", 3),
            step_retrieval_top_k=mem_cfg.get("step_retrieval_top_k", 3),
            **agent_cfg,
        )
        obs, task = env.reset(i)
        agent.reset(obs, task_type=task.task_type)
        candidates = env.get_available_actions()
        action = agent.act(candidates=candidates)

        done = False
        episode_steps = 0
        info = {}
        while not done:
            obs, reward, done, info = env.step(action)
            episode_steps += 1
            if done:
                agent.observe(obs, reward=reward)
                break
            candidates = env.get_available_actions()
            action = agent.act(obs, candidates=candidates, reward=reward)

        success = bool(info.get("goal_reached", False))
        agent.on_episode_end(success)
        total_calls += llm.total_calls
        llm.close()

        logger.save_episode(i, agent.get_episode_trace())

        if success:
            success_count += 1
        type_stats[task.task_type]["total"] += 1
        type_stats[task.task_type]["steps"].append(episode_steps)
        if success:
            type_stats[task.task_type]["success"] += 1

        results.append({
            "task_idx": i,
            "task_type": task.task_type,
            "task_name": task.task_name,
            "success": success,
            "steps": episode_steps,
            "llm_calls": llm.total_calls,
        })

        sr = success_count / (i + 1)
        print(f"  [{i+1}/{n_episodes}] {task.task_type}: "
              f"{'OK' if success else 'FAIL'} ({episode_steps} steps) | "
              f"SR={sr:.1%}", flush=True)

    elapsed = time.time() - start_time
    all_steps = [r["steps"] for r in results]
    success_steps = [r["steps"] for r in results if r["success"]]
    per_type = {}
    for task_type, stats in type_stats.items():
        total = stats["total"]
        per_type[task_type] = {
            "success": stats["success"],
            "total": total,
            "sr": stats["success"] / total if total else 0,
            "avg_steps": sum(stats["steps"]) / len(stats["steps"]) if stats["steps"] else 0,
        }

    summary = {
        "method": f"seam_ablate_{variant}",
        "ablation_variant": variant,
        "ablation_display": ABLATIONS[variant]["display"],
        "environment": "alfworld",
        "split": config["env"].get("split", ""),
        "model": model,
        "max_steps": steps_limit,
        "num_episodes": n_episodes,
        "success_rate": success_count / n_episodes,
        "success_count": success_count,
        "total_time": elapsed,
        "llm_calls": total_calls,
        "avg_steps_all": sum(all_steps) / len(all_steps) if all_steps else 0,
        "avg_steps_success": (
            sum(success_steps) / len(success_steps) if success_steps else 0
        ),
        "per_type": per_type,
        "ablation_config": config["ablation"],
        "graph_final": graph.get_stats(),
        "evolution_final": evolution.get_stats(),
        "results": results,
    }

    logger.save_memory(graph, evolution)
    logger.finalize(summary)
    if hasattr(env, "close"):
        env.close()

    print(f"\nSeam Ablation ({variant}): "
          f"{success_count}/{n_episodes} ({success_count/n_episodes:.1%})")
    print(f"Avg steps: all={summary['avg_steps_all']:.1f}, "
          f"success={summary['avg_steps_success']:.1f}")
    print(f"LLM calls: {total_calls}, Time: {elapsed:.0f}s")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("variant", nargs="?", default="all")
    parser.add_argument("num_episodes", nargs="?", type=int, default=0)
    parser.add_argument("max_steps", nargs="?", type=int, default=0)
    parser.add_argument("config_path", nargs="?", default="configs/seam_aaai2026.yaml")
    args = parser.parse_args()

    variants = list(ABLATIONS)
    if args.variant != "all":
        variants = [canonical_variant(args.variant)]
    for variant in variants:
        run_ablation(
            variant,
            config_path=args.config_path,
            num_episodes=args.num_episodes or None,
            max_steps=args.max_steps or None,
        )
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
