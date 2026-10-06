"""A-Mem baseline."""

from __future__ import annotations

from typing import Any

from agent.baselines.react import ReActAgent
from agent.baselines.utils import parse_json_object, shorten, tokenize


class AMemAgent(ReActAgent):
    """A-Mem: structured agentic memory notes with links and evolution."""

    method_name = "amem"

    def reset(self, task_obs: str, task_type: str = ""):
        self._used_note_ids: list[int] = []
        super().reset(task_obs, task_type=task_type)

    def _method_instruction(self) -> str:
        return (
            "Use A-Mem style agentic memory: consult relevant structured notes "
            "and their linked neighbors, but choose only one valid environment "
            "action for the current state."
        )

    def _initial_memory_context(self) -> str:
        return self._amem_context(self._task_obs)

    def act(self, observation: str | None = None, candidates: list[str] | None = None,
            reward: float = 0.0) -> str:
        if observation is not None and len(self._history) > 2:
            self.observe(observation, reward=reward)
            note_context = self._amem_context(
                f"{self._task_obs}\n{observation}\n{' '.join(self._actions_taken[-5:])}"
            )
            msg = self._format_step_input(observation, candidates)
            if note_context:
                msg += "\n\n" + note_context
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
                "amem_context": note_context or None,
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

    def _amem_context(self, query: str) -> str:
        if not self.memory:
            return ""
        notes = self.memory.retrieve_amem_notes(query, self._task_type, k=5, include_links=True)
        if not notes:
            return ""
        for note in notes:
            note_id = note.get("id")
            if note_id is not None and note_id not in self._used_note_ids:
                self._used_note_ids.append(note_id)
        return "[Agentic memory notes]\n" + self.memory.format_amem_notes(notes)

    def _update_memory_on_episode_end(self, success: bool,
                                      experience: dict[str, Any]) -> dict[str, Any]:
        if not self.memory:
            return {}
        query = f"{self._task_obs} {self._task_type} {' '.join(self._actions_taken)}"
        related = self.memory.retrieve_amem_notes(query, self._task_type, k=6, include_links=False)

        construct_response = self._construct_note(related, success)
        note_data = self._note_from_json(parse_json_object(construct_response), success)
        note_data.update({
            "episode": self.episode,
            "task_type": self._task_type,
            "source_success": success,
        })
        new_note = self.memory.add_amem_note(note_data)

        link_response = self._link_and_evolve_note(new_note, related)
        link_data = parse_json_object(link_response)
        links_added = self._apply_links(new_note, related, link_data)
        notes_evolved = self._apply_evolution(new_note, related, link_data)

        self._episode_trace.append({
            "type": "memory_update",
            "kind": "amem_note",
            "construct_response": construct_response,
            "link_response": link_response,
            "new_note": new_note,
            "related_note_ids": [note.get("id") for note in related],
            "links_added": links_added,
            "notes_evolved": notes_evolved,
        })
        return {
            "amem_notes_added": 1,
            "amem_links_added": links_added,
            "amem_notes_evolved": notes_evolved,
        }

    def _construct_note(self, related: list[dict[str, Any]], success: bool) -> str:
        related_text = self.memory.format_amem_notes(related) if self.memory and related else "(none)"
        result = "success" if success else "failure"
        prompt = [
            {"role": "system", "content": (
                "You implement A-Mem, an agentic memory system inspired by "
                "Zettelkasten. Convert one interaction trajectory into a structured "
                "memory note. Return strict JSON with keys: content, context, "
                "keywords, tags. content should preserve reusable procedural and "
                "outcome information. context should describe when the note applies. "
                "keywords and tags must be arrays of short strings."
            )},
            {"role": "user", "content": (
                f"Nearby existing notes:\n{related_text}\n\n"
                f"Task type: {self._task_type}\n"
                f"Task: {self._task_obs}\n"
                f"Result: {result}\n"
                f"Trajectory:\n{self._trajectory_text()}\n\n"
                "Create one memory note."
            )},
        ]
        return self.llm.chat(prompt, max_tokens=700 if self.llm.thinking_disabled else 1200)

    def _link_and_evolve_note(self, new_note: dict[str, Any],
                              related: list[dict[str, Any]]) -> str:
        if not related:
            return "{}"
        related_text = self.memory.format_amem_notes(related) if self.memory else "(none)"
        prompt = [
            {"role": "system", "content": (
                "You implement A-Mem link generation and memory evolution. Given "
                "a new note and nearby old notes, create meaningful note links and "
                "revise old notes only when the new evidence improves or corrects "
                "them. Return strict JSON with keys: links and updates. links is "
                "an array of {target_id, relation, reason}. updates is an array of "
                "{target_id, content, keywords, tags, reason}; omit fields that "
                "should not change."
            )},
            {"role": "user", "content": (
                f"New note:\n{self.memory.format_amem_notes([new_note])}\n\n"
                f"Nearby old notes:\n{related_text}\n\n"
                "Generate links and any necessary memory evolution updates."
            )},
        ]
        return self.llm.chat(prompt, max_tokens=700 if self.llm.thinking_disabled else 1200)

    def _note_from_json(self, data: dict[str, Any], success: bool) -> dict[str, Any]:
        content = data.get("content") or self._fallback_content(success)
        context = data.get("context") or self._task_obs
        keywords = data.get("keywords")
        if not isinstance(keywords, list) or not keywords:
            keywords = tokenize(f"{self._task_obs} {self._task_type} {' '.join(self._actions_taken)}")[:12]
        tags = data.get("tags")
        if not isinstance(tags, list) or not tags:
            tags = [self._task_type or "task", "success" if success else "failure"]
        return {
            "content": shorten(str(content), 900),
            "context": shorten(str(context), 360),
            "keywords": [shorten(str(item), 80) for item in keywords if str(item).strip()][:12],
            "tags": [shorten(str(item), 80) for item in tags if str(item).strip()][:10],
            "links": [],
            "evolution_history": [],
        }

    def _apply_links(self, new_note: dict[str, Any], related: list[dict[str, Any]],
                     data: dict[str, Any]) -> int:
        if not self.memory:
            return 0
        related_ids = {note.get("id") for note in related}
        links = data.get("links")
        if not isinstance(links, list):
            links = self._fallback_links(new_note, related)
        added = 0
        for link in links:
            if not isinstance(link, dict):
                continue
            try:
                target_id = int(link.get("target_id"))
            except (TypeError, ValueError):
                continue
            if target_id not in related_ids:
                continue
            relation = str(link.get("relation") or "related")
            reason = str(link.get("reason") or "")
            if self.memory.add_amem_link(new_note["id"], target_id, relation, reason):
                added += 1
            self.memory.add_amem_link(target_id, new_note["id"], f"inverse_{relation}", reason)
        return added

    def _apply_evolution(self, new_note: dict[str, Any], related: list[dict[str, Any]],
                         data: dict[str, Any]) -> int:
        if not self.memory:
            return 0
        related_ids = {note.get("id") for note in related}
        updates = data.get("updates")
        if not isinstance(updates, list):
            return 0
        changed = 0
        for update in updates:
            if not isinstance(update, dict):
                continue
            try:
                target_id = int(update.get("target_id"))
            except (TypeError, ValueError):
                continue
            if target_id not in related_ids:
                continue
            payload: dict[str, Any] = {}
            for key in ("content", "keywords", "tags"):
                if key in update:
                    payload[key] = update[key]
            reason = update.get("reason")
            if reason:
                payload["evolution_history"] = [{
                    "episode": self.episode,
                    "new_note_id": new_note["id"],
                    "reason": shorten(str(reason), 240),
                }]
            if payload and self.memory.update_amem_note(target_id, payload):
                changed += 1
        return changed

    def _fallback_content(self, success: bool) -> str:
        result = "succeeded" if success else "failed"
        actions = " -> ".join(self._actions_taken[:15])
        return (
            f"Task {result}. Task type: {self._task_type}. "
            f"Initial task: {self._task_obs}. Actions: {actions}."
        )

    @staticmethod
    def _fallback_links(new_note: dict[str, Any],
                        related: list[dict[str, Any]]) -> list[dict[str, Any]]:
        links = []
        new_terms = set(new_note.get("keywords", [])) | set(new_note.get("tags", []))
        for note in related[:4]:
            old_terms = set(note.get("keywords", [])) | set(note.get("tags", []))
            overlap = sorted(new_terms & old_terms)
            if overlap:
                links.append({
                    "target_id": note.get("id"),
                    "relation": "shares_keywords",
                    "reason": "overlap: " + ", ".join(overlap[:5]),
                })
        return links
