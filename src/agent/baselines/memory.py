"""Shared cross-episode memory store for memory-based baselines."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

from agent.baselines.utils import clean_prefixed_line, shorten, text_score, tokenize


class BaselineMemory:
    """Cross-episode text store used differently by memory-based baselines.

    The object is shared across episodes within one experiment run, but each
    baseline reads/writes a different bucket: RAG uses documents, Reflexion uses
    reflections, ExpeL uses experiences plus insights, and AutoGuide uses
    state-aware guides.
    """

    def __init__(self, max_experiences: int = 1000, max_text_items: int = 250,
                 max_reflections: int = 3, max_workflows: int = 120,
                 max_amem_notes: int = 500, max_mem0_memories: int = 1000):
        self.max_experiences = max_experiences
        self.max_text_items = max_text_items
        self.max_reflections = max_reflections
        self.max_workflows = max_workflows
        self.max_amem_notes = max_amem_notes
        self.max_mem0_memories = max_mem0_memories
        self.experiences: list[dict[str, Any]] = []
        self.reflections: list[dict[str, Any]] = []
        self.insights: list[dict[str, Any]] = []
        self.guides: list[dict[str, Any]] = []
        self.documents: list[dict[str, Any]] = []
        self.workflows: list[dict[str, Any]] = []
        self.amem_notes: list[dict[str, Any]] = []
        self.mem0_memories: list[dict[str, Any]] = []
        self._seeded_doc_sources: set[str] = set()
        self._next_workflow_id = 0
        self._next_amem_note_id = 0
        self._next_mem0_id = 0

    def add_experience(self, record: dict[str, Any]) -> dict[str, Any]:
        record = dict(record)
        record["id"] = len(self.experiences)
        self.experiences.append(record)
        if len(self.experiences) > self.max_experiences:
            self.experiences = self.experiences[-self.max_experiences :]
        return record

    def add_document(self, text: str, metadata: dict[str, Any] | None = None) -> dict[str, Any] | None:
        text = (text or "").strip()
        if not text:
            return None
        tokens = tokenize(text)
        if not tokens:
            return None
        doc = {
            "id": len(self.documents),
            "text": text,
            "metadata": metadata or {},
            "tokens": tokens,
            "term_counts": dict(Counter(tokens)),
        }
        self.documents.append(doc)
        return doc

    def seed_documents_once(self, source: str, docs: list[tuple[str, dict[str, Any]]]) -> int:
        if source in self._seeded_doc_sources:
            return 0
        added = 0
        for text, metadata in docs:
            doc = self.add_document(text, metadata)
            if doc is not None:
                added += 1
        self._seeded_doc_sources.add(source)
        return added

    def retrieve_documents(self, query: str, k: int = 3) -> list[dict[str, Any]]:
        q_terms = tokenize(query)
        if not q_terms or not self.documents:
            return []
        n_docs = len(self.documents)
        avg_len = sum(len(doc["tokens"]) for doc in self.documents) / max(n_docs, 1)
        df = Counter()
        for doc in self.documents:
            df.update(set(doc["tokens"]))

        k1 = 1.5
        b = 0.75
        scored = []
        for doc in self.documents:
            score = 0.0
            doc_len = len(doc["tokens"])
            counts = doc["term_counts"]
            for term in q_terms:
                tf = counts.get(term, 0)
                if tf == 0:
                    continue
                idf = math.log(1 + (n_docs - df[term] + 0.5) / (df[term] + 0.5))
                denom = tf + k1 * (1 - b + b * doc_len / max(avg_len, 1e-9))
                score += idf * (tf * (k1 + 1)) / denom
            if score > 0:
                scored.append((score, doc))
        scored.sort(key=lambda x: (-x[0], -x[1]["id"]))
        return [
            {
                "id": doc["id"],
                "score": round(score, 4),
                "text": doc["text"],
                "metadata": doc["metadata"],
            }
            for score, doc in scored[:k]
        ]

    def add_text_items(self, kind: str, lines: list[str], episode: int, task_type: str) -> list[str]:
        bucket = self._bucket(kind)
        existing = {self._norm_text(item.get("text", "")) for item in bucket}
        added = []
        for line in lines:
            text = clean_prefixed_line(line, ("reflection", "insight", "guide"))
            norm = self._norm_text(text)
            if not text or norm in existing:
                continue
            item = {
                "episode": episode,
                "task_type": task_type,
                "text": text,
            }
            if kind == "guide":
                item.update({"support": 0, "failure": 0})
            bucket.append(item)
            existing.add(norm)
            added.append(text)
        max_items = self.max_reflections if kind == "reflection" else self.max_text_items
        if len(bucket) > max_items:
            del bucket[: len(bucket) - max_items]
        return added

    def retrieve_experiences(self, query: str, task_type: str = "", k: int = 3,
                             successes_only: bool = False) -> list[dict[str, Any]]:
        scored = []
        for exp in self.experiences:
            if successes_only and not exp.get("success"):
                continue
            text = " ".join([
                exp.get("task_obs", ""),
                exp.get("task_type", ""),
                " ".join(exp.get("actions", [])),
            ])
            score = text_score(query, text, task_type, exp.get("task_type", ""))
            if exp.get("success"):
                score += 0.05
            if score > 0:
                scored.append((score, exp))
        scored.sort(key=lambda x: (-x[0], -x[1].get("id", 0)))
        return [exp for _, exp in scored[:k]]

    def retrieve_text_items(self, kind: str, query: str, task_type: str = "",
                            k: int = 6) -> list[dict[str, Any]]:
        scored = []
        for item in self._bucket(kind):
            score = text_score(query, item.get("text", ""), task_type, item.get("task_type", ""))
            if kind == "guide":
                support = item.get("support", 0)
                failure = item.get("failure", 0)
                score += (support + 1) / (support + failure + 2) * 0.2
            if score > 0:
                scored.append((score, item))
        scored.sort(key=lambda x: (-x[0], -x[1].get("episode", 0)))
        return [item for _, item in scored[:k]]

    def format_documents(self, docs: list[dict[str, Any]]) -> str:
        lines = []
        for doc in docs:
            meta = doc.get("metadata", {})
            label = meta.get("source", "doc")
            task_type = meta.get("task_type")
            suffix = f" / {task_type}" if task_type else ""
            lines.append(
                f"- [{label}{suffix}, score={doc.get('score', 0):.2f}] "
                f"{shorten(doc.get('text', ''), 650)}"
            )
        return "\n".join(lines)

    def format_experiences(self, experiences: list[dict[str, Any]]) -> str:
        lines = []
        for exp in experiences:
            status = "success" if exp.get("success") else "failure"
            actions = " -> ".join(exp.get("actions", [])[:12])
            lines.append(
                f"- [{status}] {exp.get('task_type', 'task')}: "
                f"{shorten(exp.get('task_obs', ''), 220)}\n"
                f"  actions: {shorten(actions, 420)}"
            )
        return "\n".join(lines)

    def format_text_items(self, items: list[dict[str, Any]]) -> str:
        return "\n".join(f"- {item.get('text', '')}" for item in items)

    def update_guide_feedback(self, guide_texts: list[str], success: bool) -> int:
        guide_set = {self._norm_text(text) for text in guide_texts if text}
        updated = 0
        for guide in self.guides:
            if self._norm_text(guide.get("text", "")) in guide_set:
                if success:
                    guide["support"] = guide.get("support", 0) + 1
                else:
                    guide["failure"] = guide.get("failure", 0) + 1
                updated += 1
        return updated

    def add_workflow(self, workflow: dict[str, Any]) -> dict[str, Any]:
        item = dict(workflow)
        item["id"] = item.get("id", self._next_workflow_id)
        self._next_workflow_id = max(self._next_workflow_id, int(item["id"]) + 1)
        item["title"] = shorten(str(item.get("title") or "Untitled workflow"), 120)
        item["task_pattern"] = shorten(str(item.get("task_pattern") or ""), 260)
        item["task_type"] = str(item.get("task_type") or "")
        item["steps"] = self._clean_list(item.get("steps"), limit=10, text_limit=220)
        item["constraints"] = self._clean_list(item.get("constraints"), limit=6, text_limit=220)
        item["source_episodes"] = list(item.get("source_episodes") or [])
        item["support"] = int(item.get("support", 0))
        item["failure"] = int(item.get("failure", 0))
        item["evidence_count"] = int(item.get("evidence_count", max(1, len(item["source_episodes"]))))
        self.workflows.append(item)
        if len(self.workflows) > self.max_workflows:
            self.workflows = self.workflows[-self.max_workflows :]
        return item

    def update_workflow(self, workflow_id: int, updates: dict[str, Any]) -> dict[str, Any] | None:
        workflow = self.get_workflow(workflow_id)
        if not workflow:
            return None
        if "title" in updates:
            workflow["title"] = shorten(str(updates["title"]), 120)
        if "task_pattern" in updates:
            workflow["task_pattern"] = shorten(str(updates["task_pattern"]), 260)
        if "steps" in updates:
            workflow["steps"] = self._clean_list(updates["steps"], limit=10, text_limit=220)
        if "constraints" in updates:
            workflow["constraints"] = self._clean_list(updates["constraints"], limit=6, text_limit=220)
        for key in ("support", "failure", "evidence_count"):
            if key in updates:
                workflow[key] = int(updates[key])
        if "source_episodes" in updates:
            merged = list(dict.fromkeys(workflow.get("source_episodes", []) + list(updates["source_episodes"])))
            workflow["source_episodes"] = merged[-20:]
        return workflow

    def get_workflow(self, workflow_id: int) -> dict[str, Any] | None:
        for workflow in self.workflows:
            if workflow.get("id") == workflow_id:
                return workflow
        return None

    def retrieve_workflows(self, query: str, task_type: str = "", k: int = 3) -> list[dict[str, Any]]:
        scored = []
        for workflow in self.workflows:
            text = " ".join([
                workflow.get("title", ""),
                workflow.get("task_pattern", ""),
                " ".join(workflow.get("steps", [])),
                " ".join(workflow.get("constraints", [])),
            ])
            score = text_score(query, text, task_type, workflow.get("task_type", ""))
            support = workflow.get("support", 0)
            failure = workflow.get("failure", 0)
            score += (support + 1) / (support + failure + 2) * 0.25
            score += min(workflow.get("evidence_count", 1), 5) * 0.02
            if score > 0:
                scored.append((score, workflow))
        scored.sort(key=lambda x: (-x[0], -x[1].get("support", 0), -x[1].get("id", 0)))
        return [workflow for _, workflow in scored[:k]]

    def update_workflow_feedback(self, workflow_ids: list[int], success: bool) -> int:
        updated = 0
        for workflow_id in set(workflow_ids):
            workflow = self.get_workflow(workflow_id)
            if not workflow:
                continue
            if success:
                workflow["support"] = workflow.get("support", 0) + 1
            else:
                workflow["failure"] = workflow.get("failure", 0) + 1
            updated += 1
        return updated

    def format_workflows(self, workflows: list[dict[str, Any]]) -> str:
        lines = []
        for workflow in workflows:
            header = (
                f"- [workflow#{workflow.get('id')}, support={workflow.get('support', 0)}, "
                f"failure={workflow.get('failure', 0)}] {workflow.get('title', '')}"
            )
            pattern = f"  task pattern: {workflow.get('task_pattern', '')}"
            steps = "\n".join(
                f"  {idx + 1}. {step}"
                for idx, step in enumerate(workflow.get("steps", [])[:8])
            )
            constraints = "; ".join(workflow.get("constraints", [])[:4])
            if constraints:
                lines.append(f"{header}\n{pattern}\n{steps}\n  constraints: {constraints}")
            else:
                lines.append(f"{header}\n{pattern}\n{steps}")
        return "\n".join(lines)

    def add_amem_note(self, note: dict[str, Any]) -> dict[str, Any]:
        item = dict(note)
        item["id"] = item.get("id", self._next_amem_note_id)
        self._next_amem_note_id = max(self._next_amem_note_id, int(item["id"]) + 1)
        item["content"] = shorten(str(item.get("content") or ""), 900)
        item["context"] = shorten(str(item.get("context") or ""), 360)
        item["task_type"] = str(item.get("task_type") or "")
        item["keywords"] = self._clean_list(item.get("keywords"), limit=12, text_limit=80)
        item["tags"] = self._clean_list(item.get("tags"), limit=10, text_limit=80)
        item["links"] = self._clean_note_links(item.get("links"))
        item["retrieval_count"] = int(item.get("retrieval_count", 0))
        item["evolution_history"] = list(item.get("evolution_history") or [])
        self.amem_notes.append(item)
        if len(self.amem_notes) > self.max_amem_notes:
            self.amem_notes = self.amem_notes[-self.max_amem_notes :]
            live_ids = {note["id"] for note in self.amem_notes}
            for note in self.amem_notes:
                note["links"] = [link for link in note.get("links", []) if link.get("target_id") in live_ids]
        return item

    def get_amem_note(self, note_id: int) -> dict[str, Any] | None:
        for note in self.amem_notes:
            if note.get("id") == note_id:
                return note
        return None

    def update_amem_note(self, note_id: int, updates: dict[str, Any]) -> dict[str, Any] | None:
        note = self.get_amem_note(note_id)
        if not note:
            return None
        if "content" in updates:
            note["content"] = shorten(str(updates["content"]), 900)
        if "context" in updates:
            note["context"] = shorten(str(updates["context"]), 360)
        if "keywords" in updates:
            note["keywords"] = self._clean_list(updates["keywords"], limit=12, text_limit=80)
        if "tags" in updates:
            note["tags"] = self._clean_list(updates["tags"], limit=10, text_limit=80)
        if "links" in updates:
            note["links"] = self._merge_note_links(note.get("links", []), updates["links"])
        if "evolution_history" in updates:
            history = list(note.get("evolution_history", [])) + list(updates["evolution_history"])
            note["evolution_history"] = history[-20:]
        return note

    def add_amem_link(self, source_id: int, target_id: int, relation: str, reason: str = "") -> bool:
        source = self.get_amem_note(source_id)
        if not source or not self.get_amem_note(target_id) or source_id == target_id:
            return False
        link = {
            "target_id": target_id,
            "relation": shorten(str(relation or "related"), 80),
            "reason": shorten(str(reason or ""), 180),
        }
        source["links"] = self._merge_note_links(source.get("links", []), [link])
        return True

    def retrieve_amem_notes(self, query: str, task_type: str = "", k: int = 5,
                            include_links: bool = True) -> list[dict[str, Any]]:
        scored = []
        for note in self.amem_notes:
            text = " ".join([
                note.get("content", ""),
                note.get("context", ""),
                " ".join(note.get("keywords", [])),
                " ".join(note.get("tags", [])),
            ])
            score = text_score(query, text, task_type, note.get("task_type", ""))
            score += min(note.get("retrieval_count", 0), 10) * 0.005
            if score > 0:
                scored.append((score, note))
        scored.sort(key=lambda x: (-x[0], -x[1].get("id", 0)))
        selected = [note for _, note in scored[:k]]
        if include_links:
            selected_by_id = {note["id"]: note for note in selected}
            for note in list(selected):
                for link in note.get("links", [])[:4]:
                    target = self.get_amem_note(link.get("target_id"))
                    if target and target["id"] not in selected_by_id:
                        selected_by_id[target["id"]] = target
            selected = list(selected_by_id.values())[: max(k, len(selected))]
        for note in selected:
            note["retrieval_count"] = note.get("retrieval_count", 0) + 1
        return selected

    def format_amem_notes(self, notes: list[dict[str, Any]]) -> str:
        lines = []
        for note in notes:
            keywords = ", ".join(note.get("keywords", [])[:6])
            links = ", ".join(
                f"{link.get('relation', 'related')}->{link.get('target_id')}"
                for link in note.get("links", [])[:4]
            )
            suffix = f"\n  links: {links}" if links else ""
            lines.append(
                f"- [note#{note.get('id')}] {shorten(note.get('content', ''), 420)}\n"
                f"  context: {shorten(note.get('context', ''), 180)}\n"
                f"  keywords: {keywords}{suffix}"
            )
        return "\n".join(lines)

    def add_mem0_memory(self, memory: dict[str, Any]) -> dict[str, Any]:
        item = dict(memory)
        item["id"] = item.get("id", self._next_mem0_id)
        self._next_mem0_id = max(self._next_mem0_id, int(item["id"]) + 1)
        item["memory"] = shorten(str(item.get("memory") or item.get("text") or ""), 520)
        item["task_type"] = str(item.get("task_type") or "")
        item["keywords"] = self._clean_list(item.get("keywords"), limit=10, text_limit=80)
        item["episode"] = int(item.get("episode", -1))
        item["source_success"] = bool(item.get("source_success", False))
        item["confidence"] = float(item.get("confidence", 0.5))
        item["retrieval_count"] = int(item.get("retrieval_count", 0))
        item["updated_episode"] = int(item.get("updated_episode", item["episode"]))
        if not item["memory"]:
            item["memory"] = "No reusable memory extracted."
        self.mem0_memories.append(item)
        if len(self.mem0_memories) > self.max_mem0_memories:
            self.mem0_memories = self.mem0_memories[-self.max_mem0_memories :]
        return item

    def get_mem0_memory(self, memory_id: int) -> dict[str, Any] | None:
        for memory in self.mem0_memories:
            if memory.get("id") == memory_id:
                return memory
        return None

    def update_mem0_memory(self, memory_id: int, updates: dict[str, Any]) -> dict[str, Any] | None:
        memory = self.get_mem0_memory(memory_id)
        if not memory:
            return None
        if "memory" in updates or "text" in updates:
            memory["memory"] = shorten(str(updates.get("memory") or updates.get("text") or ""), 520)
        if "keywords" in updates:
            memory["keywords"] = self._clean_list(updates["keywords"], limit=10, text_limit=80)
        if "task_type" in updates:
            memory["task_type"] = str(updates["task_type"] or "")
        if "confidence" in updates:
            try:
                memory["confidence"] = float(updates["confidence"])
            except (TypeError, ValueError):
                pass
        if "episode" in updates:
            try:
                memory["updated_episode"] = int(updates["episode"])
            except (TypeError, ValueError):
                pass
        return memory

    def delete_mem0_memory(self, memory_id: int) -> bool:
        before = len(self.mem0_memories)
        self.mem0_memories = [memory for memory in self.mem0_memories if memory.get("id") != memory_id]
        return len(self.mem0_memories) < before

    def retrieve_mem0_memories(self, query: str, task_type: str = "", k: int = 5) -> list[dict[str, Any]]:
        scored = []
        for memory in self.mem0_memories:
            text = " ".join([
                memory.get("memory", ""),
                " ".join(memory.get("keywords", [])),
            ])
            score = text_score(query, text, task_type, memory.get("task_type", ""))
            score += min(memory.get("retrieval_count", 0), 12) * 0.004
            score += max(0.0, min(float(memory.get("confidence", 0.5)), 1.0)) * 0.05
            if memory.get("source_success"):
                score += 0.03
            if score > 0:
                scored.append((score, memory))
        scored.sort(key=lambda x: (-x[0], -x[1].get("updated_episode", -1), -x[1].get("id", 0)))
        selected = [memory for _, memory in scored[:k]]
        for memory in selected:
            memory["retrieval_count"] = memory.get("retrieval_count", 0) + 1
        return selected

    def format_mem0_memories(self, memories: list[dict[str, Any]]) -> str:
        lines = []
        for memory in memories:
            keywords = ", ".join(memory.get("keywords", [])[:6])
            suffix = f" keywords: {keywords}" if keywords else ""
            lines.append(
                f"- [mem#{memory.get('id')}, confidence={memory.get('confidence', 0.5):.2f}] "
                f"{shorten(memory.get('memory', ''), 420)}{suffix}"
            )
        return "\n".join(lines)

    def stats(self) -> dict[str, int]:
        return {
            "experiences": len(self.experiences),
            "reflections": len(self.reflections),
            "insights": len(self.insights),
            "guides": len(self.guides),
            "documents": len(self.documents),
            "workflows": len(self.workflows),
            "amem_notes": len(self.amem_notes),
            "mem0_memories": len(self.mem0_memories),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "stats": self.stats(),
            "experiences": self.experiences,
            "reflections": self.reflections,
            "insights": self.insights,
            "guides": self.guides,
            "workflows": self.workflows,
            "amem_notes": self.amem_notes,
            "mem0_memories": self.mem0_memories,
            "documents": [
                {
                    "id": doc["id"],
                    "text": doc["text"],
                    "metadata": doc["metadata"],
                }
                for doc in self.documents
            ],
        }

    def _bucket(self, kind: str) -> list[dict[str, Any]]:
        if kind == "reflection":
            return self.reflections
        if kind == "insight":
            return self.insights
        if kind == "guide":
            return self.guides
        raise ValueError(f"Unknown baseline memory kind: {kind}")

    @staticmethod
    def _norm_text(text: str) -> str:
        return re.sub(r"\s+", " ", (text or "").lower()).strip()

    @staticmethod
    def _clean_list(value: Any, limit: int = 8, text_limit: int = 180) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            raw_items = re.split(r"[\n;]+", value)
        elif isinstance(value, list):
            raw_items = value
        else:
            raw_items = [value]
        cleaned = []
        seen = set()
        for item in raw_items:
            text = re.sub(r"^[-*\d.)\s]+", "", str(item or "")).strip()
            text = shorten(text, text_limit)
            norm = BaselineMemory._norm_text(text)
            if not text or norm in seen:
                continue
            cleaned.append(text)
            seen.add(norm)
            if len(cleaned) >= limit:
                break
        return cleaned

    @staticmethod
    def _clean_note_links(value: Any) -> list[dict[str, Any]]:
        if not isinstance(value, list):
            return []
        links = []
        seen = set()
        for link in value:
            if not isinstance(link, dict):
                continue
            try:
                target_id = int(link.get("target_id"))
            except (TypeError, ValueError):
                continue
            relation = shorten(str(link.get("relation") or "related"), 80)
            reason = shorten(str(link.get("reason") or ""), 180)
            key = (target_id, BaselineMemory._norm_text(relation))
            if key in seen:
                continue
            links.append({
                "target_id": target_id,
                "relation": relation,
                "reason": reason,
            })
            seen.add(key)
            if len(links) >= 12:
                break
        return links

    @staticmethod
    def _merge_note_links(existing: list[dict[str, Any]], new_links: Any) -> list[dict[str, Any]]:
        return BaselineMemory._clean_note_links(list(existing or []) + list(new_links or []))
