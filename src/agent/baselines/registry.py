"""Baseline method registry and aliases."""

from __future__ import annotations


BASELINE_ORDER = (
    "baseline",
    "react",
    "rag",
    "reflexion",
    "expel",
    "autoguide",
    "rap",
    "awm",
    "amem",
    "mem0",
    "synapse",
    "swiftsage",
)

BASELINE_MEMORY_METHODS = {
    "rag",
    "reflexion",
    "expel",
    "autoguide",
    "awm",
    "amem",
    "mem0",
    "synapse",
}

BASELINE_ALIASES = {
    "vanilla": "baseline",
    "reflection": "reflexion",
    "reflextion": "reflexion",
    "agent_workflow_memory": "awm",
    "agent-workflow-memory": "awm",
    "a-mem": "amem",
    "agentic_memory": "amem",
    "agentic-memory": "amem",
    "mem-zero": "mem0",
    "mem_zero": "mem0",
    "swift-sage": "swiftsage",
    "swift_sage": "swiftsage",
}

BASELINE_METHODS = set(BASELINE_ORDER) | set(BASELINE_ALIASES)


def canonical_baseline_method(method: str) -> str:
    key = (method or "baseline").strip().lower()
    return BASELINE_ALIASES.get(key, key)
