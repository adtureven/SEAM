"""
Sequential evaluation for Seam and baselines.
Memory starts empty and accumulates across episodes.
This demonstrates the self-evolution of causal memory from zero knowledge.
"""

import sys
import json
import time
import yaml
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from env.alfworld_pddl import ALFWorldEnv
from env.alfworld_adapter import ALFWorldAdapter
from agent.secm_agent import SEAMAgent
from agent.base_agent import BaseAgent
from agent.baseline_agents import (
    BASELINE_METHODS,
    BASELINE_MEMORY_METHODS,
    BASELINE_ORDER,
    BaselineMemory,
    canonical_baseline_method,
    make_baseline_agent,
)
from memory.causal_graph import HierarchicalCausalGraph
from memory.evolution import EvolutionEngine
from llm.client import make_llm_from_config
from utils.logger import ExperimentLogger


MODEL_SHORT_NAMES = {
    "mimo-v2.5": "mimo25",
    "qwen3.5-35b-a3b": "qwen35",
    "qwen3.5-4b": "qwen35_4b",
    "qwen3.5-9b": "qwen35_9b",
}


def _canonical_env_name(env_name: str) -> str:
    env_key = env_name.replace("-", "_")
    if env_key == "text_world":
        return "textworld"
    if env_key in {
        "textworld",
        "textworld_easy",
        "textworld_medium",
        "textworld_hard",
        "textworld_curriculum",
        "textworld_interleaved",
        "textworld_cooking",
        "textworld_cooking_easy",
        "textworld_cooking_medium",
        "textworld_cooking_hard",
        "textworld_cooking_curriculum",
        "textworld_cooking_interleaved",
        "textworld_treasure",
        "textworld_treasure_easy",
        "textworld_treasure_medium",
        "textworld_treasure_hard",
        "textworld_treasure_curriculum",
        "textworld_treasure_interleaved",
        "webshop",
    }:
        return env_key
    return "scienceworld" if env_name == "sciworld" else env_name


def _is_textworld_env(env_name: str) -> bool:
    return _canonical_env_name(env_name).startswith("textworld")


def _is_webshop_env(env_name: str) -> bool:
    return _canonical_env_name(env_name) == "webshop"


def _textworld_overrides(env_name: str, config: dict | None = None) -> dict:
    env_name = _canonical_env_name(env_name)
    tw_base = (config or {}).get("textworld", {})
    suite = ""
    games_dir = None
    if env_name.startswith("textworld_cooking"):
        suite = "cooking"
        games_dir = tw_base.get("cooking_games_dir", "data/textworld_cooking/games")
    elif env_name.startswith("textworld_treasure"):
        suite = "treasure"
        games_dir = tw_base.get("treasure_games_dir", "data/textworld_treasure/games")

    overrides = {}
    if suite:
        overrides["suite_name"] = suite
    if games_dir:
        overrides["games_dir"] = games_dir

    short_name = env_name
    if suite:
        short_name = env_name.replace(f"textworld_{suite}", "textworld", 1)

    if short_name == "textworld_easy":
        overrides.update({"difficulties": ["easy"], "order": "grouped", "num_tasks": 100})
    elif short_name == "textworld_medium":
        overrides.update({"difficulties": ["medium"], "order": "grouped", "num_tasks": 100})
    elif short_name == "textworld_hard":
        overrides.update({"difficulties": ["hard"], "order": "grouped", "num_tasks": 100})
    elif short_name == "textworld_curriculum":
        overrides.update({"difficulties": ["easy", "medium", "hard"], "order": "grouped", "num_tasks": 300})
    elif short_name == "textworld_interleaved":
        overrides.update({"difficulties": ["easy", "medium", "hard"], "order": "interleaved", "num_tasks": 300})
    elif short_name == "textworld" and suite:
        overrides.update({"difficulties": ["easy", "medium", "hard"], "order": "interleaved", "num_tasks": 300})
    return overrides


def _env_split(config: dict, env_name: str) -> str:
    env_name = _canonical_env_name(env_name)
    if env_name == "scienceworld":
        return (config.get("scienceworld") or config.get("sciworld", {})).get("split", "test")
    if env_name.startswith("textworld"):
        return config.get("textworld", {}).get("split", "generated")
    if env_name == "webshop":
        return config.get("webshop", {}).get("split", "all")
    return config["env"].get("split", "")


def _make_llm(config, log_path=None):
    return make_llm_from_config(config, log_path=log_path)


def _create_env_and_adapter(config, max_steps, env_name="alfworld"):
    env_name = _canonical_env_name(env_name)
    if env_name == "alfworld":
        env = ALFWorldEnv(
            data_path=config["env"]["data_path"],
            split=config["env"]["split"],
            max_steps=max_steps,
        )
        adapter = ALFWorldAdapter()
        return env, adapter
    elif env_name == "scienceworld":
        from env.scienceworld_env import ScienceWorldEnv
        from env.scienceworld_adapter import ScienceWorldAdapter
        sw_cfg = config.get("scienceworld") or config.get("sciworld", {})
        task_indices = sw_cfg.get("task_indices")
        if task_indices is None and sw_cfg.get("task_indices_file"):
            with open(sw_cfg["task_indices_file"], encoding="utf-8") as f:
                index_data = json.load(f)
            if isinstance(index_data, dict) and "task_indices" in index_data:
                task_indices = index_data["task_indices"]
            else:
                grouped = index_data.get("groups", index_data)
                if isinstance(grouped, dict):
                    task_indices = []
                    for values in grouped.values():
                        task_indices.extend(values)
                elif isinstance(grouped, list):
                    task_indices = grouped
            if task_indices is None and isinstance(index_data, dict):
                task_indices = []
                for row in index_data.get("episodes", []):
                    if "task_idx" in row:
                        task_indices.append(row["task_idx"])
        env = ScienceWorldEnv(
            task_names=sw_cfg.get("task_names"),
            split=sw_cfg.get("split", "test"),
            variation_idx=sw_cfg.get("variation_idx", 0),
            simplification=sw_cfg.get("simplification", "easy"),
            jar_path=sw_cfg.get("jar_path"),
            expand_variations=sw_cfg.get("expand_variations", False),
            task_indices=task_indices,
            max_steps=max_steps,
        )
        adapter = ScienceWorldAdapter()
        return env, adapter
    elif env_name.startswith("textworld"):
        from env.textworld_env import TextWorldEnv
        from env.textworld_adapter import TextWorldAdapter
        tw_cfg = dict(config.get("textworld", {}))
        tw_cfg.update(_textworld_overrides(env_name, config))
        env = TextWorldEnv(
            games_dir=tw_cfg.get("games_dir", "data/textworld/games"),
            difficulties=tw_cfg.get("difficulties", ["easy", "medium", "hard"]),
            num_tasks=tw_cfg.get("num_tasks"),
            home_dir=tw_cfg.get("home_dir", "data/textworld/.home"),
            max_steps=max_steps,
            order=tw_cfg.get("order", "interleaved"),
            suite_name=tw_cfg.get("suite_name", ""),
        )
        adapter = TextWorldAdapter()
        return env, adapter
    elif env_name == "webshop":
        from env.webshop_env import WebShopEnv
        from env.webshop_adapter import WebShopAdapter
        ws_cfg = config.get("webshop", {})
        env = WebShopEnv(
            webshop_repo=ws_cfg.get("repo_path"),
            observation_mode=ws_cfg.get("observation_mode", "text"),
            num_products=ws_cfg.get("num_products"),
            limit_goals=ws_cfg.get("limit_goals", -1),
            human_goals=ws_cfg.get("human_goals", False),
            show_attrs=ws_cfg.get("show_attrs", False),
            split=ws_cfg.get("split", "all"),
            max_steps=max_steps,
        )
        adapter = WebShopAdapter()
        return env, adapter
    else:
        raise ValueError(f"Unknown environment: {env_name}")


def run_cold_start(config, num_episodes, max_steps=50, env_name="alfworld"):
    env_name = _canonical_env_name(env_name)
    config["_env_name"] = env_name
    model = config["llm"]["model"]
    model_short = MODEL_SHORT_NAMES.get(model, model.split("/")[-1][:10])
    logger = ExperimentLogger(
        method=f"seam_cold_{env_name}", model_short=model_short,
        config=config, base_dir=config["experiment"]["log_dir"],
    )

    from agent.secm_agent import _load_react_prompts
    react_prompts = _load_react_prompts()
    logger.save_prompts(
        system_prompt="[environment-specific system prompt]",
        few_shot_examples=react_prompts if react_prompts else {},
    )

    print(f"=== Seam Cold Start [{env_name}] (max_steps={max_steps}) ===")
    print(f"Output: {logger.run_dir}")
    start = time.time()

    env, adapter = _create_env_and_adapter(config, max_steps, env_name)

    mem_cfg = config["memory"]
    graph = HierarchicalCausalGraph(
        stability_threshold=mem_cfg["stability_threshold"],
        stable_reliability=mem_cfg.get("stable_reliability", 0.7),
        revision_error_threshold=mem_cfg.get("revision_error_threshold", 0.5),
        aggregation_threshold=mem_cfg.get("aggregation_threshold", 2),
        adapter=adapter,
    )
    evolution = EvolutionEngine(
        graph=graph,
        adapter=adapter,
        aggregation_threshold=mem_cfg.get("aggregation_threshold", 2),
        abstraction_min_members=mem_cfg.get("abstraction_min_members", 2),
        abstraction_min_confidence=mem_cfg.get("abstraction_min_confidence", 0.6),
        revision_error_threshold=mem_cfg.get("revision_error_threshold", 0.5),
    )

    results = []
    success_count = 0
    type_stats = defaultdict(
        lambda: {"success": 0, "total": 0, "steps": [], "scores": [], "normalized_scores": []}
    )

    num_episodes = min(num_episodes, env.num_tasks())

    for i in range(num_episodes):
        llm = _make_llm(config, log_path=logger.llm_log_path)
        agent = SEAMAgent(
            llm,
            graph,
            evolution,
            adapter=adapter,
            episode=i,
            enable_llm_reflection=mem_cfg.get("enable_llm_reflection", False),
            enable_effect_prediction=mem_cfg.get("enable_effect_prediction", True),
            retrieval_top_k=mem_cfg.get("retrieval_top_k", 3),
            step_retrieval_top_k=mem_cfg.get("step_retrieval_top_k", 3),
        )
        obs, task = env.reset(i)
        agent.reset(obs, task_type=task.task_type)
        candidates = env.get_available_actions()
        action = agent.act(candidates=candidates)

        done = False
        steps = 0
        while not done:
            obs, reward, done, info = env.step(action)
            steps += 1
            if done:
                agent.observe(obs, reward=reward)
                break
            candidates = env.get_available_actions()
            action = agent.act(obs, candidates=candidates, reward=reward)

        success = info.get("goal_reached", False)
        agent.on_episode_end(success)
        llm.close()

        logger.save_episode(i, agent.get_episode_trace())

        if success:
            success_count += 1

        sr = success_count / (i + 1)
        stats = graph.get_stats()
        metrics = _score_metrics(info)
        type_stats[task.task_type]["total"] += 1
        type_stats[task.task_type]["steps"].append(steps)
        if metrics["eval_score"] is not None:
            type_stats[task.task_type]["scores"].append(metrics["eval_score"])
        if metrics["normalized_score"] is not None:
            type_stats[task.task_type]["normalized_scores"].append(metrics["normalized_score"])
        if success:
            type_stats[task.task_type]["success"] += 1

        result = {
            "task_idx": i,
            "task_type": task.task_type,
            "task_name": task.task_name,
            "task_desc": task.task_desc,
            "success": success,
            "steps": steps,
            "llm_calls": llm.total_calls,
            "memory_stats": stats,
        }
        result.update(metrics)
        results.append(result)

        score_text = (
            f" | score={metrics['normalized_score']:.2f}"
            if metrics["normalized_score"] is not None else ""
        )
        print(f"  [{i+1}/{num_episodes}] {task.task_type}: "
              f"{'OK' if success else 'FAIL'} ({steps} steps) | "
              f"SR={sr:.1%}{score_text} | M0={stats['m0_hypothesis']} | "
              f"M1={stats['m1_stable']} "
              f"(stable={stats['m1_belief_stable']}) | "
              f"M2={stats['m2_abstract']} "
              f"(stable={stats['m2_belief_stable']})", flush=True)

    elapsed = time.time() - start
    total_calls = sum(r["llm_calls"] for r in results)

    n = len(results)
    early = results[:n//4]
    middle = results[n//4:3*n//4]
    late = results[3*n//4:]
    early_sr = sum(1 for r in early if r["success"]) / len(early) if early else 0
    middle_sr = sum(1 for r in middle if r["success"]) / len(middle) if middle else 0
    late_sr = sum(1 for r in late if r["success"]) / len(late) if late else 0

    type_summary = {}
    for ttype, ts in type_stats.items():
        type_summary[ttype] = {
            "success": ts["success"],
            "total": ts["total"],
            "sr": ts["success"] / ts["total"] if ts["total"] > 0 else 0,
            "avg_steps": sum(ts["steps"]) / len(ts["steps"]) if ts["steps"] else 0,
            "avg_score": sum(ts["scores"]) / len(ts["scores"]) if ts["scores"] else None,
            "avg_normalized_score": (
                sum(ts["normalized_scores"]) / len(ts["normalized_scores"])
                if ts["normalized_scores"] else None
            ),
        }

    all_steps = [r["steps"] for r in results]
    success_steps = [r["steps"] for r in results if r["success"]]
    normalized_scores = [
        r["normalized_score"] for r in results
        if r.get("normalized_score") is not None
    ]
    scores = [
        r["eval_score"] for r in results
        if r.get("eval_score") is not None
    ]

    summary = {
        "method": f"seam_cold_{env_name}",
        "environment": env_name,
        "split": _env_split(config, env_name),
        "model": config["llm"]["model"],
        "max_steps": max_steps,
        "num_episodes": num_episodes,
        "success_rate": success_count / num_episodes,
        "success_count": success_count,
        "total_time": elapsed,
        "llm_calls": total_calls,
        "avg_steps_all": sum(all_steps) / len(all_steps),
        "avg_steps_success": sum(success_steps) / len(success_steps) if success_steps else 0,
        "avg_score": sum(scores) / len(scores) if scores else None,
        "avg_normalized_score": (
            sum(normalized_scores) / len(normalized_scores)
            if normalized_scores else None
        ),
        "learning_curve": {
            "early_sr": early_sr,
            "middle_sr": middle_sr,
            "late_sr": late_sr,
        },
        "per_type": type_summary,
        "final_memory": graph.get_stats(),
        "evolution_stats": evolution.get_stats(),
        "results": results,
    }

    logger.save_memory(graph, evolution)
    logger.finalize(summary)

    print(f"\n{'='*60}")
    print(f"Seam Cold Start [{env_name}] (max_steps={max_steps}): "
          f"{success_count}/{num_episodes} ({success_count/num_episodes:.1%})")
    print(f"Learning curve: early={early_sr:.1%}, middle={middle_sr:.1%}, late={late_sr:.1%}")
    avg_s = sum(success_steps)/len(success_steps) if success_steps else 0
    print(f"Avg steps: all={sum(all_steps)/len(all_steps):.1f}, success={avg_s:.1f}")
    print(f"LLM calls: {total_calls}, Time: {elapsed:.0f}s")
    print(f"\nPer-type SR:")
    for ttype in sorted(type_summary):
        ts = type_summary[ttype]
        if ts.get("avg_score") is not None:
            score_suffix = f" avg_score={ts['avg_score']:.1f}"
        elif ts.get("avg_normalized_score") is not None:
            score_suffix = f" avg_score={ts['avg_normalized_score']:.2f}"
        else:
            score_suffix = ""
        print(f"  {ttype}: {ts['success']}/{ts['total']} ({ts['sr']:.0%}) "
              f"avg_steps={ts['avg_steps']:.1f}{score_suffix}")
    print(f"\nFinal memory: {graph.get_stats()}")
    if hasattr(env, "close"):
        env.close()
    return summary


def _agent_reset(agent, obs: str, task):
    if agent.__class__ is BaseAgent:
        agent.reset(obs)
    else:
        agent.reset(obs, task_type=task.task_type)


def _agent_act(agent, observation=None, candidates=None, reward: float = 0.0):
    if agent.__class__ is BaseAgent:
        return agent.act(observation, candidates=candidates)
    return agent.act(observation, candidates=candidates, reward=reward)


def _agent_observe(agent, observation: str, reward: float = 0.0):
    if hasattr(agent, "observe") and agent.__class__ is not BaseAgent:
        agent.observe(observation, reward=reward)


def _score_metrics(info: dict) -> dict:
    score = info.get("score")
    eval_score = info.get("score_for_eval")
    max_score = info.get("max_score")
    normalized_score = info.get("normalized_score")
    try:
        score = float(score) if score is not None else None
    except (TypeError, ValueError):
        score = None
    try:
        eval_score = float(eval_score) if eval_score is not None else None
    except (TypeError, ValueError):
        eval_score = None
    if eval_score is None:
        eval_score = score
    try:
        max_score = float(max_score) if max_score is not None else None
    except (TypeError, ValueError):
        max_score = None
    try:
        normalized_score = float(normalized_score) if normalized_score is not None else None
    except (TypeError, ValueError):
        normalized_score = None
    if normalized_score is None and eval_score is not None and max_score and max_score > 0:
        normalized_score = eval_score / max_score
    elif (
        normalized_score is None
        and eval_score is not None
        and "taskName" in info
        and 0 <= eval_score <= 100
    ):
        normalized_score = eval_score / 100.0
    return {
        "score": score,
        "eval_score": eval_score,
        "max_score": max_score,
        "normalized_score": normalized_score,
    }


def run_baseline(config, num_episodes, max_steps=50, env_name="alfworld",
                 baseline_method: str = "baseline"):
    env_name = _canonical_env_name(env_name)
    config["_env_name"] = env_name
    baseline_method = canonical_baseline_method(baseline_method)
    if baseline_method not in BASELINE_ORDER:
        raise ValueError(f"Unknown baseline method: {baseline_method}")
    model = config["llm"]["model"]
    model_short = MODEL_SHORT_NAMES.get(model, model.split("/")[-1][:10])
    logger = ExperimentLogger(
        method=f"{baseline_method}_{env_name}", model_short=model_short,
        config=config, base_dir=config["experiment"]["log_dir"],
    )

    print(f"=== {baseline_method.upper()} [{env_name}] (max_steps={max_steps}) ===")
    print(f"Output: {logger.run_dir}")
    start = time.time()

    env, adapter = _create_env_and_adapter(config, max_steps, env_name)
    baseline_memory = BaselineMemory() if baseline_method in BASELINE_MEMORY_METHODS else None

    results = []
    success_count = 0
    type_stats = defaultdict(
        lambda: {"success": 0, "total": 0, "steps": [], "scores": [], "normalized_scores": []}
    )

    num_episodes = min(num_episodes, env.num_tasks())
    baseline_cfg = config.get("baseline", {})
    reflexion_max_trials = int(baseline_cfg.get("reflexion_max_trials", 3))

    for i in range(num_episodes):
        max_trials = reflexion_max_trials if baseline_method == "reflexion" else 1
        episode_trace = []
        task = None
        info = {}
        steps = 0
        trial_count = 0
        task_llm_calls = 0
        success = False

        for trial_idx in range(max_trials):
            trial_count += 1
            llm = _make_llm(config, log_path=logger.llm_log_path)
            agent = make_baseline_agent(
                baseline_method, llm, adapter=adapter,
                memory=baseline_memory, episode=i,
                baseline_config=baseline_cfg,
            )
            obs, task = env.reset(i)
            _agent_reset(agent, obs, task)
            candidates = env.get_available_actions()
            action = _agent_act(agent, candidates=candidates)

            done = False
            trial_steps = 0
            while not done:
                obs, reward, done, info = env.step(action)
                trial_steps += 1
                if done:
                    _agent_observe(agent, obs, reward=reward)
                    break
                candidates = env.get_available_actions()
                action = _agent_act(agent, obs, candidates=candidates, reward=reward)

            trial_success = info.get("goal_reached", False)
            agent.on_episode_end(trial_success)
            task_llm_calls += llm.total_calls
            llm.close()

            episode_trace.append({
                "type": "trial",
                "trial": trial_idx + 1,
                "success": trial_success,
                "steps": trial_steps,
            })
            for entry in agent.get_episode_trace():
                item = dict(entry)
                item["trial"] = trial_idx + 1
                episode_trace.append(item)

            steps += trial_steps
            success = trial_success
            if success:
                break

        logger.save_episode(i, episode_trace)

        if success:
            success_count += 1

        sr = success_count / (i + 1)
        metrics = _score_metrics(info)
        type_stats[task.task_type]["total"] += 1
        type_stats[task.task_type]["steps"].append(steps)
        if metrics["eval_score"] is not None:
            type_stats[task.task_type]["scores"].append(metrics["eval_score"])
        if metrics["normalized_score"] is not None:
            type_stats[task.task_type]["normalized_scores"].append(metrics["normalized_score"])
        if success:
            type_stats[task.task_type]["success"] += 1

        result = {
            "task_idx": i,
            "task_type": task.task_type,
            "task_name": task.task_name,
            "task_desc": task.task_desc,
            "success": success,
            "steps": steps,
            "trials": trial_count,
            "llm_calls": task_llm_calls,
            "memory_stats": baseline_memory.stats() if baseline_memory else {},
        }
        result.update(metrics)
        results.append(result)

        trial_text = f", {trial_count} trials" if trial_count > 1 else ""
        score_text = (
            f" | score={metrics['normalized_score']:.2f}"
            if metrics["normalized_score"] is not None else ""
        )
        print(f"  [{i+1}/{num_episodes}] {task.task_type}: "
              f"{'OK' if success else 'FAIL'} ({steps} steps{trial_text}) | "
              f"SR={sr:.1%}{score_text}", flush=True)

    elapsed = time.time() - start
    total_calls = sum(r["llm_calls"] for r in results)
    all_steps = [r["steps"] for r in results]
    success_steps = [r["steps"] for r in results if r["success"]]
    normalized_scores = [
        r["normalized_score"] for r in results
        if r.get("normalized_score") is not None
    ]
    scores = [
        r["eval_score"] for r in results
        if r.get("eval_score") is not None
    ]

    type_summary = {}
    for ttype, ts in type_stats.items():
        type_summary[ttype] = {
            "success": ts["success"],
            "total": ts["total"],
            "sr": ts["success"] / ts["total"] if ts["total"] > 0 else 0,
            "avg_steps": sum(ts["steps"]) / len(ts["steps"]) if ts["steps"] else 0,
            "avg_score": sum(ts["scores"]) / len(ts["scores"]) if ts["scores"] else None,
            "avg_normalized_score": (
                sum(ts["normalized_scores"]) / len(ts["normalized_scores"])
                if ts["normalized_scores"] else None
            ),
        }

    summary = {
        "method": f"{baseline_method}_{env_name}",
        "baseline_method": baseline_method,
        "environment": env_name,
        "split": _env_split(config, env_name),
        "model": config["llm"]["model"],
        "max_steps": max_steps,
        "num_episodes": num_episodes,
        "success_rate": success_count / num_episodes,
        "success_count": success_count,
        "total_time": elapsed,
        "llm_calls": total_calls,
        "avg_steps_all": sum(all_steps) / len(all_steps),
        "avg_steps_success": sum(success_steps) / len(success_steps) if success_steps else 0,
        "avg_score": sum(scores) / len(scores) if scores else None,
        "avg_normalized_score": (
            sum(normalized_scores) / len(normalized_scores)
            if normalized_scores else None
        ),
        "per_type": type_summary,
        "baseline_memory": baseline_memory.stats() if baseline_memory else {},
        "results": results,
    }

    if baseline_memory:
        with open(logger.run_dir / "memory" / f"{baseline_method}_memory.json", "w") as f:
            json.dump(baseline_memory.to_dict(), f, indent=2, ensure_ascii=False)
    logger.finalize(summary)

    print(f"\n{'='*60}")
    print(f"{baseline_method.upper()} [{env_name}] (max_steps={max_steps}): "
          f"{success_count}/{num_episodes} ({success_count/num_episodes:.1%})")
    avg_s_bl = sum(success_steps)/len(success_steps) if success_steps else 0
    print(f"Avg steps: all={sum(all_steps)/len(all_steps):.1f}, success={avg_s_bl:.1f}")
    print(f"LLM calls: {total_calls}, Time: {elapsed:.0f}s")
    print(f"\nPer-type SR:")
    for ttype in sorted(type_summary):
        ts = type_summary[ttype]
        if ts.get("avg_score") is not None:
            score_suffix = f" avg_score={ts['avg_score']:.1f}"
        elif ts.get("avg_normalized_score") is not None:
            score_suffix = f" avg_score={ts['avg_normalized_score']:.2f}"
        else:
            score_suffix = ""
        print(f"  {ttype}: {ts['success']}/{ts['total']} ({ts['sr']:.0%}) "
              f"avg_steps={ts['avg_steps']:.1f}{score_suffix}")
    if hasattr(env, "close"):
        env.close()
    return summary


def main():
    # Parse args
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("num_episodes", type=int, nargs="?", default=0)
    parser.add_argument("method", nargs="?", default="seam")
    parser.add_argument("max_steps", type=int, nargs="?", default=0)
    parser.add_argument("config_path", nargs="?", default="configs/seam_aaai2026.yaml")
    parser.add_argument("--env", default="alfworld", choices=[
        "alfworld", "scienceworld", "sciworld",
        "textworld", "text-world", "textworld_easy", "textworld_medium",
        "textworld_hard", "textworld_curriculum", "textworld_interleaved",
        "textworld_cooking", "textworld_cooking_easy", "textworld_cooking_medium",
        "textworld_cooking_hard", "textworld_cooking_curriculum", "textworld_cooking_interleaved",
        "textworld_treasure", "textworld_treasure_easy", "textworld_treasure_medium",
        "textworld_treasure_hard", "textworld_treasure_curriculum", "textworld_treasure_interleaved",
        "webshop",
    ])
    args = parser.parse_args()

    with open(args.config_path) as f:
        config = yaml.safe_load(f)

    num_episodes = args.num_episodes or config["experiment"]["num_episodes"]
    env_name = _canonical_env_name(args.env)
    if _is_webshop_env(env_name) and not args.num_episodes:
        num_episodes = int(config.get("webshop", {}).get("num_episodes", 500))
    if args.max_steps:
        max_steps = args.max_steps
    elif _is_textworld_env(env_name):
        max_steps = int(config.get("textworld", {}).get("max_steps", config["env"]["max_steps"]))
    elif _is_webshop_env(env_name):
        max_steps = int(config.get("webshop", {}).get("max_steps", config["env"]["max_steps"]))
    else:
        max_steps = config["env"]["max_steps"]
    method = args.method.strip().lower()

    if method in ("baselines", "all_baselines"):
        for baseline_method in BASELINE_ORDER:
            try:
                run_baseline(
                    config, num_episodes, max_steps=max_steps,
                    env_name=args.env, baseline_method=baseline_method,
                )
            except ImportError as exc:
                raise SystemExit(f"Dependency error: {exc}") from None
            print()
        return

    if method == "both":
        try:
            run_baseline(
                config, num_episodes, max_steps=max_steps,
                env_name=args.env, baseline_method="baseline",
            )
        except ImportError as exc:
            raise SystemExit(f"Dependency error: {exc}") from None
        print()

    if method in BASELINE_METHODS and method != "baseline":
        try:
            run_baseline(
                config, num_episodes, max_steps=max_steps,
                env_name=args.env, baseline_method=method,
            )
        except ImportError as exc:
            raise SystemExit(f"Dependency error: {exc}") from None
        print()
        return

    if method == "baseline":
        try:
            run_baseline(
                config, num_episodes, max_steps=max_steps,
                env_name=args.env, baseline_method="baseline",
            )
        except ImportError as exc:
            raise SystemExit(f"Dependency error: {exc}") from None
        print()
        return

    if method in ("both", "seam", "secm"):
        try:
            run_cold_start(config, num_episodes, max_steps=max_steps, env_name=args.env)
        except ImportError as exc:
            raise SystemExit(f"Dependency error: {exc}") from None
        print()
        return

    allowed = ", ".join(list(BASELINE_ORDER) + ["baselines", "both", "seam", "secm"])
    raise SystemExit(f"Unknown method: {args.method}. Expected one of: {allowed}")


if __name__ == "__main__":
    main()
