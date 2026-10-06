"""Shared text utilities for baseline agents."""

from __future__ import annotations

import json
import re
from pathlib import Path


_REACT_PROMPT_PATH = Path(__file__).parent.parent.parent.parent / "data" / "react_prompts.json"
_REACT_PROMPTS: dict[str, str] = {}

_STOPWORDS = {
    "the", "and", "for", "you", "are", "with", "that", "this", "from",
    "task", "your", "some", "then", "into", "onto", "have", "need",
    "using", "there", "here", "room", "action", "actions",
}


def tokenize(text: str) -> list[str]:
    return [
        w for w in re.findall(r"[a-z0-9_]+", (text or "").lower())
        if len(w) >= 3 and w not in _STOPWORDS
    ]


def load_react_prompts() -> dict[str, str]:
    global _REACT_PROMPTS
    if not _REACT_PROMPTS and _REACT_PROMPT_PATH.exists():
        with open(_REACT_PROMPT_PATH, encoding="utf-8") as f:
            _REACT_PROMPTS = json.load(f)
    return _REACT_PROMPTS


def keywords(text: str) -> set[str]:
    return set(tokenize(text))


def text_score(query: str, text: str, task_type: str = "", item_task_type: str = "") -> float:
    q = keywords(query)
    t = keywords(text)
    score = 0.0
    if q and t:
        score += len(q & t) / max(len(q), 1)
    if task_type and item_task_type and task_type == item_task_type:
        score += 0.35
    return score


def shorten(text: str, limit: int = 700) -> str:
    text = re.sub(r"\s+", " ", (text or "")).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def clean_prefixed_line(line: str, prefixes: tuple[str, ...]) -> str:
    line = line.strip()
    line = re.sub(r"^[-*]\s*", "", line)
    line = re.sub(r"^\d+[\.)]\s*", "", line)
    for prefix in prefixes:
        pattern = rf"(?i)^{re.escape(prefix)}\s*:?\s*"
        line = re.sub(pattern, "", line).strip()
    return line


def parse_prefixed_lines(response: str, prefixes: tuple[str, ...], limit: int = 5) -> list[str]:
    items = []
    for raw in (response or "").splitlines():
        line = clean_prefixed_line(raw, prefixes)
        if not line:
            continue
        if len(line.split()) < 3:
            continue
        if len(line) > 260:
            line = shorten(line, 260)
        items.append(line)
        if len(items) >= limit:
            break
    return items


def parse_json_object(response: str) -> dict:
    text = (response or "").strip()
    if not text:
        return {}
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        return {}
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}
