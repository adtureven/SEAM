"""SwiftSage baseline."""

from __future__ import annotations

import re
from typing import Any

from agent.baselines.react import ReActAgent
from agent.baselines.utils import parse_json_object, shorten


class SwiftSageAgent(ReActAgent):
    """SwiftSage: fast action proposal with selective slow deliberation."""

    method_name = "swiftsage"

    def __init__(self, *args, confidence_threshold: float = 0.62,
                 repeat_window: int = 4, **kwargs):
        super().__init__(*args, **kwargs)
        self.confidence_threshold = confidence_threshold
        self.repeat_window = repeat_window

    def _method_instruction(self) -> str:
        return (
            "Use SwiftSage-style dual-process control. First make a swift "
            "intuitive action proposal. When the proposal is uncertain, invalid, "
            "or the recent trajectory looks stuck, use a slower deliberative "
            "check before acting."
        )

    def act(self, observation: str | None = None, candidates: list[str] | None = None,
            reward: float = 0.0) -> str:
        if observation is not None and len(self._history) > 2:
            self.observe(observation, reward=reward)

        step_input = self._format_step_input(observation, candidates)
        if step_input and (observation is not None or candidates):
            self._history.append({"role": "user", "content": step_input})
        self._trim_history()

        call_idx_before = self.llm.total_calls
        fast_response = self._fast_policy(observation, candidates)
        fast_action, confidence, rationale = self._parse_fast_response(fast_response)
        needs_slow, reasons = self._needs_slow_check(
            fast_action, confidence, candidates, observation
        )

        slow_response = ""
        action = fast_action
        if needs_slow:
            slow_response = self._slow_policy(observation, candidates, fast_action, rationale, reasons)
            action = self._parse_candidate_action(slow_response, candidates) if candidates else self._parse_action(slow_response)
        if candidates:
            action = self._validate_action(action, candidates)
        self._actions_taken.append(action)
        self._history.append({"role": "assistant", "content": f"Action: {action}"})

        self._episode_trace.append({
            "type": "step",
            "step": len(self._actions_taken),
            "observation": observation,
            "candidates_count": len(candidates) if candidates else 0,
            "candidates_sample": candidates[:20] if candidates else [],
            "action": action,
            "swift_sage": {
                "fast_response": fast_response.strip(),
                "fast_action": fast_action,
                "confidence": confidence,
                "rationale": rationale,
                "slow_used": needs_slow,
                "slow_reasons": reasons,
                "slow_response": slow_response.strip() or None,
            },
            "llm_calls": [{
                "call_idx_start": call_idx_before + 1,
                "call_idx_end": self.llm.total_calls,
                "num_calls": self.llm.total_calls - call_idx_before,
                "parsed_action": action,
            }],
        })
        return action

    def _fast_policy(self, observation: str | None, candidates: list[str] | None) -> str:
        prompt_candidates = self._last_prompt_candidates or self._select_prompt_candidates(observation, candidates or [])
        candidates_text = "\n".join(f"- {item}" for item in prompt_candidates) or "(not provided)"
        if candidates and len(prompt_candidates) < len(candidates):
            candidates_text = f"(showing {len(prompt_candidates)} of {len(candidates)} most relevant)\n{candidates_text}"
        prompt = [
            {"role": "system", "content": (
                "You are the Swift module in SwiftSage: a fast intuitive policy "
                "for text-game agents. Choose one action quickly. Return strict "
                "JSON with keys: action, confidence, rationale. confidence must "
                "be a number from 0 to 1."
            )},
            {"role": "user", "content": (
                f"Task type: {self._task_type}\n"
                f"Task: {self._task_obs}\n"
                f"Recent real trajectory:\n{self._trajectory_text()}\n\n"
                f"Current observation:\n{observation or self._last_observation}\n\n"
                f"Available actions:\n{candidates_text}\n\n"
                "Return one swift action proposal."
            )},
        ]
        return self.llm.chat(prompt, max_tokens=220 if self.llm.thinking_disabled else 500)

    def _slow_policy(self, observation: str | None, candidates: list[str] | None,
                     fast_action: str, rationale: str, reasons: list[str]) -> str:
        prompt_candidates = self._last_prompt_candidates or self._select_prompt_candidates(observation, candidates or [])
        candidates_text = "\n".join(f"- {item}" for item in prompt_candidates) or "(not provided)"
        if candidates and len(prompt_candidates) < len(candidates):
            candidates_text = f"(showing {len(prompt_candidates)} of {len(candidates)} most relevant)\n{candidates_text}"
        prompt = [
            {"role": "system", "content": (
                "You are the Sage module in SwiftSage: a deliberate reasoner that "
                "checks the fast policy, diagnoses uncertainty or stagnation, and "
                "selects one robust next environment action. The final line must be "
                "exactly: Action: <one valid action>."
            )},
            {"role": "user", "content": (
                f"Task type: {self._task_type}\n"
                f"Task: {self._task_obs}\n"
                f"Recent real trajectory:\n{self._trajectory_text()}\n\n"
                f"Current observation:\n{observation or self._last_observation}\n\n"
                f"Fast proposal: {fast_action}\n"
                f"Fast rationale: {rationale}\n"
                f"Reasons for slow check: {', '.join(reasons)}\n\n"
                f"Available actions:\n{candidates_text}"
            )},
        ]
        return self.llm.chat(prompt, max_tokens=450 if self.llm.thinking_disabled else 900)

    def _parse_fast_response(self, response: str) -> tuple[str, float, str]:
        data = parse_json_object(response)
        if data:
            action = str(data.get("action") or "").strip().lower()
            confidence = self._clip01(data.get("confidence", 0.0))
            rationale = shorten(str(data.get("rationale") or ""), 240)
            if action:
                return action, confidence, rationale
        action = self._parse_action(response)
        confidence = self._extract_confidence(response)
        return action, confidence, shorten(response, 240)

    def _needs_slow_check(self, action: str, confidence: float, candidates: list[str] | None,
                          observation: str | None) -> tuple[bool, list[str]]:
        reasons = []
        if confidence < self.confidence_threshold:
            reasons.append("low_confidence")
        if candidates and action not in candidates:
            action_low = action.lower().strip()
            if not any(c.lower().strip() == action_low for c in candidates):
                reasons.append("invalid_fast_action")
        if self._recently_repeated(action):
            reasons.append("repeated_action")
        if self._looks_stuck(observation):
            reasons.append("stagnation_signal")
        return bool(reasons), reasons

    def _recently_repeated(self, action: str) -> bool:
        if not action or len(self._actions_taken) < 2:
            return False
        recent = [item.lower().strip() for item in self._actions_taken[-self.repeat_window:]]
        return recent.count(action.lower().strip()) >= 2

    @staticmethod
    def _looks_stuck(observation: str | None) -> bool:
        text = (observation or "").lower()
        signals = (
            "nothing happens",
            "can't",
            "cannot",
            "not possible",
            "invalid",
            "don't understand",
            "you already",
            "there is no",
            "you need to",
        )
        return any(signal in text for signal in signals)

    @staticmethod
    def _extract_confidence(text: str) -> float:
        match = re.search(r"(?i)confidence\s*[:=]\s*([0-9.]+)", text or "")
        if not match:
            return 0.5
        return SwiftSageAgent._clip01(match.group(1))

    @staticmethod
    def _clip01(value: Any) -> float:
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return 0.0
