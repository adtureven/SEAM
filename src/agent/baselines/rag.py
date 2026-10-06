"""Retrieval-augmented generation baseline."""

from __future__ import annotations

from typing import Any

from agent.baselines.react import ReActAgent
from agent.baselines.utils import load_react_prompts


class RAGAgent(ReActAgent):
    """RAG baseline: retrieve prior trajectories as non-parametric memory."""

    method_name = "rag"
    retrieval_k = 4

    def reset(self, task_obs: str, task_type: str = ""):
        if self.memory:
            self._seed_static_corpus()
        super().reset(task_obs, task_type=task_type)

    def _method_instruction(self) -> str:
        return (
            "Use retrieval-augmented generation: consult the retrieved prior "
            "experience snippets when they are relevant, then choose one action."
        )

    def _initial_memory_context(self) -> str:
        return self._retrieval_context(self._task_obs)

    def act(self, observation: str | None = None, candidates: list[str] | None = None,
            reward: float = 0.0) -> str:
        if observation is not None and len(self._history) > 2:
            self.observe(observation, reward=reward)
            retrieval_context = self._retrieval_context(
                f"{self._task_obs}\n{observation}\n{' '.join(self._actions_taken[-5:])}"
            )
            msg = self._format_step_input(observation, candidates)
            if retrieval_context:
                msg += "\n\n" + retrieval_context
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
                "retrieval_context": retrieval_context or None,
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

    def _retrieval_context(self, query: str) -> str:
        if not self.memory:
            return ""
        docs = self.memory.retrieve_documents(query, k=self.retrieval_k)
        if not docs:
            return ""
        return "[Retrieved passages]\n" + self.memory.format_documents(docs)

    def _update_memory_on_episode_end(self, success: bool,
                                      experience: dict[str, Any]) -> dict[str, Any]:
        if not self.memory:
            return {}
        text = (
            f"Task type: {self._task_type}\n"
            f"Task: {self._task_obs}\n"
            f"Result: {'success' if success else 'failure'}\n"
            f"Trajectory:\n{self._trajectory_text()}"
        )
        doc = self.memory.add_document(text, {
            "source": "episode",
            "episode": self.episode,
            "task_type": self._task_type,
            "success": success,
        })
        return {"documents_added": 1 if doc else 0}

    def _seed_static_corpus(self):
        prompts = load_react_prompts()
        docs: list[tuple[str, dict[str, Any]]] = []
        for key, text in prompts.items():
            if not key.startswith(("react_", "act_")):
                continue
            parts = key.split("_")
            task_type = parts[1] if len(parts) > 1 else ""
            docs.append((
                text,
                {
                    "source": "react_prompt",
                    "prompt_key": key,
                    "task_type": task_type,
                },
            ))
        self.memory.seed_documents_once("react_prompts", docs)
