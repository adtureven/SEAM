"""Factory for baseline agents."""

from __future__ import annotations

from agent.base_agent import BaseAgent
from agent.baselines.amem import AMemAgent
from agent.baselines.autoguide import AutoGuideAgent
from agent.baselines.awm import AWMAgent
from agent.baselines.expel import ExpeLAgent
from agent.baselines.memory import BaselineMemory
from agent.baselines.mem0 import Mem0Agent
from agent.baselines.rag import RAGAgent
from agent.baselines.rap import RAPAgent
from agent.baselines.react import ReActAgent
from agent.baselines.reflexion import ReflexionAgent
from agent.baselines.registry import canonical_baseline_method
from agent.baselines.swiftsage import SwiftSageAgent
from agent.baselines.synapse import SynapseAgent
from env.adapter import EnvironmentAdapter
from llm.client import LLMClient


def make_baseline_agent(method: str, llm: LLMClient, adapter: EnvironmentAdapter,
                        memory: BaselineMemory | None = None, episode: int = 0,
                        baseline_config: dict | None = None):
    method = canonical_baseline_method(method)
    if method == "baseline":
        return BaseAgent(llm, adapter=adapter)
    if method == "react":
        return ReActAgent(llm, adapter=adapter, memory=None, episode=episode)
    if method == "rag":
        return RAGAgent(llm, adapter=adapter, memory=memory, episode=episode)
    if method == "reflexion":
        return ReflexionAgent(llm, adapter=adapter, memory=memory, episode=episode)
    if method == "expel":
        return ExpeLAgent(llm, adapter=adapter, memory=memory, episode=episode)
    if method == "autoguide":
        return AutoGuideAgent(llm, adapter=adapter, memory=memory, episode=episode)
    if method == "rap":
        rap_cfg = (baseline_config or {}).get("rap", {})
        return RAPAgent(
            llm,
            adapter=adapter,
            memory=None,
            episode=episode,
            max_history_steps=int(rap_cfg.get("max_history_steps", 6)),
            max_candidates=int(rap_cfg.get("max_candidates", 8)),
            rollout_depth=int(rap_cfg.get("rollout_depth", 2)),
            branch_width=int(rap_cfg.get("branch_width", 3)),
            world_model_max_tokens=int(rap_cfg.get("world_model_max_tokens", 512)),
        )
    if method == "awm":
        return AWMAgent(llm, adapter=adapter, memory=memory, episode=episode)
    if method == "amem":
        return AMemAgent(llm, adapter=adapter, memory=memory, episode=episode)
    if method == "mem0":
        return Mem0Agent(llm, adapter=adapter, memory=memory, episode=episode)
    if method == "synapse":
        return SynapseAgent(llm, adapter=adapter, memory=memory, episode=episode)
    if method == "swiftsage":
        return SwiftSageAgent(llm, adapter=adapter, memory=None, episode=episode)
    raise ValueError(f"Unknown baseline method: {method}")
