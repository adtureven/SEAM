"""Mem0-style long-term memory baseline."""

from __future__ import annotations

from typing import Any

from agent.baselines.react import ReActAgent
from agent.baselines.utils import parse_json_object, shorten, tokenize


class Mem0Agent(ReActAgent):
    """Mem0 baseline: extract, update, delete, and retrieve salient memories."""

    method_name = "mem0"

    def reset(self, task_obs: str, task_type: str = ""):
        self._used_memory_ids: list[int] = []
        super().reset(task_obs, task_type=task_type)

    def _method_instruction(self) -> str:
        return (
            "Use Mem0-style long-term memory: retrieve concise relevant memories "
            "from previous episodes, apply them only when they match the current "
            "task, and choose exactly one valid environment action."
        )

    def _initial_memory_context(self) -> str:
        return self._mem0_context(self._task_obs)

    def act(self, observation: str | None = None, candidates: list[str] | None = None,
            reward: float = 0.0) -> str:
        if observation is not None and len(self._history) > 2:
            self.observe(observation, reward=reward)
            memory_context = self._mem0_context(
                f"{self._task_obs}\n{observation}\n{' '.join(self._actions_taken[-5:])}"
            )
            msg = self._format_step_input(observation, candidates)
            if memory_context:
                msg += "\n\n" + memory_context
            self._history.append({"role": "user", "content": msg})

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
                "mem0_context": memory_context or None,
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

        return super().act(observation, candidates=candidates, reward=reward)

    def _mem0_context(self, query: str) -> str:
        if not self.memory:
            return ""
        memories = self.memory.retrieve_mem0_memories(query, self._task_type, k=6)
        if not memories:
            return ""
        for memory in memories:
            memory_id = memory.get("id")
            if memory_id is not None and memory_id not in self._used_memory_ids:
                self._used_memory_ids.append(memory_id)
        return "[Mem0 memories]\n" + self.memory.format_mem0_memories(memories)

    def _update_memory_on_episode_end(self, success: bool,
                                      experience: dict[str, Any]) -> dict[str, Any]:
        if not self.memory:
            return {}
        query = f"{self._task_obs} {self._task_type} {' '.join(self._actions_taken)}"
        related = self.memory.retrieve_mem0_memories(query, self._task_type, k=8)
        response = self._propose_memory_operations(related, success)
        data = parse_json_object(response)
        operations = data.get("operations")
        if not isinstance(operations, list):
            operations = self._fallback_operations(success)

        related_ids = [memory.get("id") for memory in related if memory.get("id") is not None]
        stats = self._apply_operations(operations, related_ids, success)
        self._episode_trace.append({
            "type": "memory_update",
            "kind": "mem0_operations",
            "response": response,
            "related_memory_ids": related_ids,
            "applied": stats,
        })
        return stats

    def _propose_memory_operations(self, related: list[dict[str, Any]], success: bool) -> str:
        related_text = self.memory.format_mem0_memories(related) if self.memory and related else "(none)"
        result = "success" if success else "failure"
        prompt = [
            {"role": "system", "content": (
                "You implement Mem0, a long-term memory layer for agents. "
                "From a new interaction, extract only durable, reusable facts or "
                "procedures. Compare them with retrieved memories and return strict "
                "JSON with key operations. Each operation must be one of: "
                "ADD, UPDATE, DELETE, NOOP. Use ADD for new reusable information, "
                "UPDATE when a retrieved memory should be corrected or merged, "
                "DELETE when a retrieved memory is contradicted or harmful, and "
                "NOOP when nothing durable should change. Schema: "
                "{\"operations\":[{\"event\":\"ADD|UPDATE|DELETE|NOOP\","
                "\"id\":0,\"memory\":\"text\",\"keywords\":[\"short\"],"
                "\"confidence\":0.0,\"reason\":\"text\"}]}. "
                "For ADD, omit id. For UPDATE/DELETE, id must refer to a retrieved memory."
            )},
            {"role": "user", "content": (
                f"Retrieved memories:\n{related_text}\n\n"
                f"Task type: {self._task_type}\n"
                f"Task: {self._task_obs}\n"
                f"Result: {result}\n"
                f"Trajectory:\n{self._trajectory_text()}\n\n"
                "Generate memory operations. Prefer one to three concise memories."
            )},
        ]
        return self.llm.chat(prompt, max_tokens=700 if self.llm.thinking_disabled else 1200)

    def _apply_operations(self, operations: list[Any], related_ids: list[int],
                          success: bool) -> dict[str, int]:
        stats = {"added": 0, "updated": 0, "deleted": 0, "noop": 0, "invalid": 0}
        valid_related_ids = set(related_ids) | set(self._used_memory_ids)
        for op in operations[:6]:
            if not isinstance(op, dict):
                stats["invalid"] += 1
                continue
            event = str(op.get("event") or op.get("op") or "NOOP").strip().upper()
            if event not in {"ADD", "UPDATE", "DELETE", "NOOP"}:
                stats["invalid"] += 1
                continue
            if event == "NOOP":
                stats["noop"] += 1
                continue

            if event == "ADD":
                memory_text = str(op.get("memory") or op.get("text") or "").strip()
                if not memory_text:
                    stats["invalid"] += 1
                    continue
                self.memory.add_mem0_memory(self._memory_payload(op, memory_text, success))
                stats["added"] += 1
                continue

            try:
                memory_id = int(op.get("id"))
            except (TypeError, ValueError):
                stats["invalid"] += 1
                continue
            if valid_related_ids and memory_id not in valid_related_ids:
                stats["invalid"] += 1
                continue
            if event == "UPDATE":
                memory_text = str(op.get("memory") or op.get("text") or "").strip()
                payload = self._memory_payload(op, memory_text, success) if memory_text else {
                    "keywords": self._keywords_from_op(op),
                    "confidence": self._confidence_from_op(op),
                    "episode": self.episode,
                }
                if self.memory.update_mem0_memory(memory_id, payload):
                    stats["updated"] += 1
                else:
                    stats["invalid"] += 1
            elif event == "DELETE":
                if self.memory.delete_mem0_memory(memory_id):
                    stats["deleted"] += 1
                else:
                    stats["invalid"] += 1
        return stats

    def _memory_payload(self, op: dict[str, Any], memory_text: str,
                        success: bool) -> dict[str, Any]:
        return {
            "memory": shorten(memory_text, 520),
            "task_type": self._task_type,
            "keywords": self._keywords_from_op(op),
            "episode": self.episode,
            "source_success": success,
            "confidence": self._confidence_from_op(op),
        }

    def _keywords_from_op(self, op: dict[str, Any]) -> list[str]:
        keywords = op.get("keywords")
        if not isinstance(keywords, list) or not keywords:
            keywords = tokenize(f"{self._task_obs} {self._task_type} {' '.join(self._actions_taken)}")[:10]
        return [shorten(str(item), 80) for item in keywords if str(item).strip()][:10]

    @staticmethod
    def _confidence_from_op(op: dict[str, Any]) -> float:
        try:
            return max(0.0, min(float(op.get("confidence", 0.65)), 1.0))
        except (TypeError, ValueError):
            return 0.65

    def _fallback_operations(self, success: bool) -> list[dict[str, Any]]:
        result = "succeeded" if success else "failed"
        actions = " -> ".join(self._actions_taken[:14])
        if not actions:
            return [{"event": "NOOP"}]
        return [{
            "event": "ADD",
            "memory": (
                f"For {self._task_type or 'this task type'}, an episode {result}. "
                f"Task: {shorten(self._task_obs, 180)}. Actions: {shorten(actions, 260)}."
            ),
            "keywords": tokenize(f"{self._task_obs} {self._task_type} {actions}")[:10],
            "confidence": 0.55 if success else 0.35,
            "reason": "fallback from trajectory because Mem0 JSON parsing failed",
        }]
