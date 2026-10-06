"""
Base Agent: vanilla LLM agent (no memory).
Observes environment, generates actions via LLM.
"""

import re
import time
from llm.client import LLMClient
from env.adapter import EnvironmentAdapter


ALFWORLD_SYSTEM_PROMPT = """You are an agent solving household tasks in a text-based environment.
You can interact with the environment using these actions:
- look: observe current location
- inventory: check what you're holding
- go to <receptacle>: move to a receptacle
- open <receptacle>: open a container
- close <receptacle>: close a container
- take <object> from <receptacle>: pick up an object
- put <object> in/on <receptacle>: place the object you're holding
- heat <object> with <microwave>: heat an object
- clean <object> with <sink/sinkbasin>: clean an object
- cool <object> with <fridge>: cool an object
- use <object>: toggle a device on/off
- examine <object/receptacle>: look closely at something

Rules:
- You can only hold one object at a time
- Some receptacles need to be opened before you can access objects inside
- Respond with ONLY the action you want to take, nothing else"""


class BaseAgent:
    MAX_PROMPT_CANDIDATES = 80
    MAX_PROMPT_CANDIDATE_CHARS = 5000

    def __init__(self, llm: LLMClient, adapter: EnvironmentAdapter | None = None):
        self.llm = llm
        self.adapter = adapter
        self._history: list[dict] = []
        self._episode_trace: list[dict] = []
        self._actions_taken: list[str] = []
        self._start_time: float = 0
        self._last_prompt_candidates: list[str] = []

    def _get_system_prompt(self) -> str:
        if self.adapter:
            base = self.adapter.get_system_prompt()
            action_text = self.adapter.get_action_list_text()
            return (f"{base}\n\n{action_text}\n\n"
                    "Rules:\n"
                    "- Respond with ONLY the action you want to take, nothing else.")
        return ALFWORLD_SYSTEM_PROMPT

    def reset(self, task_obs: str):
        system_prompt = self._get_system_prompt()
        self._history = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task_obs},
        ]
        self._episode_trace = [{
            "type": "init",
            "system_prompt": system_prompt,
            "task_obs": task_obs,
        }]
        self._actions_taken = []
        self._start_time = time.time()

    def act(self, observation: str | None = None, candidates: list[str] | None = None) -> str:
        user_msg = self._format_step_input(observation, candidates)
        if user_msg and (observation is not None or candidates):
            self._history.append({"role": "user", "content": user_msg})
        call_idx_before = self.llm.total_calls
        response = self.llm.chat(self._history)
        action = self._parse_candidate_action(response, candidates) if candidates else self._parse_action(response)
        if candidates:
            action = self._validate_action(action, candidates)
        self._actions_taken.append(action)
        self._history.append({"role": "assistant", "content": action})

        self._episode_trace.append({
            "type": "step",
            "step": len(self._actions_taken),
            "observation": observation,
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

    def _format_step_input(self, observation: str | None,
                           candidates: list[str] | None = None) -> str:
        parts = []
        if observation is not None:
            parts.append(observation)
        if candidates:
            prompt_candidates = self._select_prompt_candidates(observation, candidates)
            actions = "\n".join(f"- {c}" for c in prompt_candidates)
            label = "Available actions for this step"
            if len(prompt_candidates) < len(candidates):
                label += f" (showing {len(prompt_candidates)} of {len(candidates)} most relevant)"
            has_search_template = any(
                c.strip().lower() == "search[query]" for c in prompt_candidates
            )
            choice_rule = (
                "For search[query], replace query with useful product keywords. "
                "Copy click actions from this list exactly."
                if has_search_template else "Choose exactly one action from this list."
            )
            parts.append(
                f"{label}:\n"
                f"{actions}\n"
                f"{choice_rule}"
            )
        return "\n\n".join(parts)

    def on_episode_end(self, success: bool):
        self._episode_trace.append({
            "type": "result",
            "success": success,
            "total_steps": len(self._actions_taken),
            "total_llm_calls": self.llm.total_calls,
            "duration_seconds": round(time.time() - self._start_time, 2),
        })

    def get_episode_trace(self) -> list[dict]:
        return self._episode_trace

    def _validate_action(self, action: str, candidates: list[str]) -> str:
        action = self._normalize_action(action, candidates)
        if action in candidates:
            return action
        action_low = action.lower().strip()
        for c in candidates:
            if c.lower().strip() == action_low:
                return c
        if self.adapter and self.adapter.allow_unlisted_action(action, candidates):
            return action
        for c in candidates:
            template = c.lower().strip()
            if self._matches_template_candidate(action_low, template):
                return action
        for c in candidates:
            if action_low in c.lower() or c.lower() in action_low:
                return c
        if self._last_prompt_candidates:
            return self._last_prompt_candidates[0]
        return candidates[0] if candidates else action

    def _parse_candidate_action(self, response: str, candidates: list[str]) -> str:
        lines = [line.strip() for line in (response or "").splitlines() if line.strip()]
        for line in reversed(lines):
            clean = self._clean_action_line(line)
            if not clean:
                continue
            clean = self._normalize_action(clean, candidates).lower().strip()
            for candidate in candidates:
                if clean == candidate.lower().strip():
                    return candidate
            for candidate in candidates:
                c_low = candidate.lower().strip()
                if clean in c_low or c_low in clean:
                    return candidate
            if not clean.startswith(("thought:", "think:", "plan:", "observation:", "reason:")):
                return clean
        return self._parse_action(response)

    def _select_prompt_candidates(self, observation: str | None,
                                  candidates: list[str]) -> list[str]:
        if not candidates:
            self._last_prompt_candidates = []
            return []
        if self._candidate_text_len(candidates) <= self.MAX_PROMPT_CANDIDATE_CHARS:
            selected = list(candidates[: self.MAX_PROMPT_CANDIDATES])
            self._last_prompt_candidates = selected
            return selected

        query = " ".join([
            self._history[1]["content"] if len(self._history) > 1 else "",
            observation or "",
            " ".join(self._actions_taken[-5:]),
        ])
        query_terms = set(re.findall(r"[a-z0-9]+", query.lower()))
        recent = {a.lower().strip() for a in self._actions_taken[-4:]}
        scored = []
        for idx, action in enumerate(candidates):
            low = action.lower().strip()
            terms = set(re.findall(r"[a-z0-9]+", low))
            overlap = len(query_terms & terms)
            score = overlap * 3 + self._candidate_command_score(low)
            if low in recent:
                score -= 6
            if low.startswith("focus on "):
                score -= 3
            if low.startswith("close "):
                score -= 1
            score -= min(len(low) // 120, 4)
            scored.append((score, idx, action))
        scored.sort(key=lambda item: (-item[0], item[1]))

        selected = []
        total_chars = 0
        for _, _, action in scored:
            added = len(action) + 3
            if selected and total_chars + added > self.MAX_PROMPT_CANDIDATE_CHARS:
                break
            selected.append(action)
            total_chars += added
            if len(selected) >= self.MAX_PROMPT_CANDIDATES:
                break
        if not selected:
            selected = list(candidates[:1])
        self._last_prompt_candidates = selected
        return selected

    @staticmethod
    def _candidate_text_len(candidates: list[str]) -> int:
        return sum(len(item) + 3 for item in candidates)

    @staticmethod
    def _candidate_command_score(action: str) -> int:
        priorities = (
            (("take ", "pick up ", "put ", "place ", "drop "), 10),
            (("mix ", "dunk ", "heat ", "cool ", "measure ", "use ", "activate ", "turn on ",
              "prepare ", "cook ", "slice ", "dice ", "chop "), 9),
            (("open ",), 7),
            (("go to ", "move to ", "go "), 6),
            (("examine ", "look"), 4),
            (("read ", "eat "), 3),
            (("inventory",), 2),
            (("focus on ",), 1),
            (("close ",), 0),
        )
        for prefixes, score in priorities:
            if action.startswith(prefixes):
                return score
        return 1

    @staticmethod
    def _clean_action_line(line: str) -> str:
        line = line.strip()
        if line.startswith(">"):
            line = line[1:].strip()
        line = re.sub(r"(?i)^(final\s+)?action\s*:\s*", "", line).strip()
        line = re.sub(r"^\d+[\.)]\s*", "", line).strip()
        return line.lower()

    @staticmethod
    def _matches_template_candidate(action: str, template: str) -> bool:
        if "<" not in template or ">" not in template:
            return False
        prefix = template.split("<", 1)[0]
        suffix = template.split(">", 1)[1]
        return action.startswith(prefix) and action.endswith(suffix)

    def _parse_action(self, response: str) -> str:
        response = response.strip().lower()
        if self.adapter:
            valid_cmds = self.adapter.get_valid_commands()
        else:
            valid_cmds = ("look", "inventory", "go to", "open", "close", "take",
                          "put", "move", "heat", "clean", "cool", "use", "examine")
        lines = response.split("\n")
        for line in lines:
            line = line.strip()
            if line.startswith(">"):
                line = line[1:].strip()
            line = re.sub(r"^(action:\s*)", "", line)
            line = self._normalize_action(line)
            if any(line.startswith(cmd) for cmd in valid_cmds):
                return line
        return "look"

    def _normalize_action(self, action: str, candidates: list[str] | None = None) -> str:
        if self.adapter:
            return self.adapter.normalize_action(action, candidates)
        return action.strip()
