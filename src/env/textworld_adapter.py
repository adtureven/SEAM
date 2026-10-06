"""TextWorld adapter for Seam and baseline agents."""

from __future__ import annotations

import re

from env.adapter import EnvironmentAdapter


class TextWorldAdapter(EnvironmentAdapter):
    def parse_observation(self, obs: str) -> dict:
        low = obs.lower()
        result: dict = {}
        room_match = re.search(r"-=\s*([^=\n]+?)\s*=-", obs)
        if room_match:
            result["arrived_at"] = self._clean_name(room_match.group(1))
        if any(x in low for x in ("you can't", "you cannot", "i don't understand", "that's not a verb")):
            result["error"] = obs
        if "you are carrying" in low or "you have" in low:
            result["inventory_observed"] = True
        if any(x in low for x in ("you pick up", "you take", "taken.")):
            result["picked_up"] = True
        if any(x in low for x in ("you drop", "you put", "placed")):
            result["put_down"] = True
        if any(x in low for x in ("you open", "open the")):
            result["processed"] = "open"
        if any(x in low for x in ("you close", "close the")):
            result["processed"] = "close"
        if any(x in low for x in ("you unlock", "unlock the")):
            result["processed"] = "unlock"
        if any(x in low for x in ("you eat", "you have eaten")):
            result["processed"] = "eat"
        exits = re.findall(r"\b(?:north|south|east|west|up|down)\b", low)
        if exits:
            result["available_directions"] = sorted(set(exits))
        objects = re.findall(r"\b(?:a|an|the) ([a-z][a-z0-9 -]{1,40})", obs, flags=re.I)
        if objects:
            cleaned = [self._clean_name(obj) for obj in objects]
            filtered = [obj for obj in cleaned if self._looks_like_object(obj)]
            if filtered:
                result["objects_here"] = filtered[:20]
        return result

    def obj_name_to_type(self, name: str) -> str:
        return self._fallback_type(name)

    def recep_name_to_type(self, name: str) -> str:
        return self._fallback_type(name)

    def parse_task(self, obs: str, task_type: str = "") -> dict:
        desc = ""
        match = re.search(r"(?:objective|goal)\s*:\s*(.+)", obs, flags=re.I | re.S)
        if match:
            desc = re.sub(r"\s+", " ", match.group(1)).strip()
            desc = desc.split("Score:", 1)[0].strip()
        else:
            desc = self._extract_inline_objective(obs)
        target = self._extract_target(desc)
        return {
            "task_type": task_type or "textworld",
            "target_obj": target,
            "target_obj_type": self.obj_name_to_type(target) if target else "",
            "target_recep": "",
            "processing": self._extract_processing(desc),
        }

    def action_matches_goto(self, action: str) -> str | None:
        match = re.match(r"(?:go\s+)?(north|south|east|west|up|down)\b", action.strip(), flags=re.I)
        return match.group(1).lower() if match else None

    def action_matches_take(self, action: str, target_obj: str) -> bool:
        low = action.lower().strip()
        if not low.startswith(("take ", "get ", "pick up ")):
            return False
        return bool(target_obj) and target_obj.lower() in low

    def action_matches_process(self, action: str, process_type: str) -> bool:
        low = action.lower().strip()
        return bool(process_type) and low.startswith(process_type.lower() + " ")

    def action_matches_put(self, action: str) -> bool:
        return action.lower().strip().startswith(("drop ", "put ", "insert "))

    def action_matches_use_device(self, action: str) -> bool:
        return action.lower().strip().startswith((
            "open ", "close ", "unlock ", "lock ", "eat ", "read ", "examine ", "look",
            "prepare ", "cook ", "slice ", "dice ", "chop ",
        ))

    def normalize_action(self, action: str, candidates: list[str] | None = None) -> str:
        low = re.sub(r"\s+", " ", (action or "").strip().lower())
        low = low.strip(" .,:;")
        low = re.sub(r"^(?:final\s+)?action\s*:\s*", "", low).strip()

        replacements = (
            (r"^look\s+(?:at|in|inside|into)\s+(.+)$", r"examine \1"),
            (r"^inspect\s+(.+)$", r"examine \1"),
            (r"^check\s+(.+)$", r"examine \1"),
            (r"^read\s+(.+)$", r"examine \1"),
            (r"^open\s+(?:the\s+)?(?:recipe|cookbook)$", "examine cookbook"),
            (r"^look\s+(?:at\s+)?(?:the\s+)?recipe$", "examine cookbook"),
            (r"^read\s+(?:the\s+)?recipe$", "examine cookbook"),
            (r"^prepare\s+(?:the\s+)?meal$", "prepare meal"),
            (r"^make\s+(?:the\s+)?meal$", "prepare meal"),
            (r"^eat\s+(?:the\s+)?meal$", "eat meal"),
        )
        for pattern, repl in replacements:
            if re.search(pattern, low):
                low = re.sub(pattern, repl, low)
                break

        low = re.sub(r"^(examine|take|drop|open|close|unlock|lock|eat|prepare|cook|slice|dice|chop)\s+(?:the|a|an)\s+", r"\1 ", low)
        if candidates:
            match = self._match_candidate(low, candidates)
            if match:
                return match
            if any(token in low for token in ("cookbook", "recipe")):
                for candidate in candidates:
                    c_low = candidate.lower().strip()
                    if c_low.startswith("examine ") and "cookbook" in c_low:
                        return candidate
            if "meal" in low:
                preferred = "prepare " if low.startswith(("prepare ", "make ", "cook ")) else "eat "
                for candidate in candidates:
                    c_low = candidate.lower().strip()
                    if c_low.startswith(preferred) and "meal" in c_low:
                        return candidate
        return low

    def allow_unlisted_action(self, action: str, candidates: list[str] | None = None) -> bool:
        low = action.lower().strip()
        if low in ("look", "inventory", "north", "south", "east", "west", "up", "down"):
            return True
        if low.startswith("go "):
            return bool(re.match(r"^go\s+(north|south|east|west|up|down)$", low))
        return low.startswith(("examine ", "read "))

    def extract_action_effect(self, action: str) -> tuple[str, str] | None:
        low = action.lower().strip()
        for verb in (
            "open", "close", "unlock", "lock", "eat", "read", "examine", "prepare",
            "cook", "slice", "dice", "chop", "take", "drop", "put", "insert",
        ):
            if low.startswith(verb + " "):
                return verb, self._object_from_action(low, verb)
        if low in ("look", "inventory") or self.action_matches_goto(low):
            return (low.split()[0], "")
        return None

    def infer_appliance_type(self, appliance_name: str) -> str:
        return self._fallback_type(appliance_name)

    def get_object_category(self, obj_type: str) -> str | None:
        return "textworld_object" if obj_type else None

    def get_receptacle_category(self, recep_type: str) -> str | None:
        return "textworld_location" if recep_type else None

    def get_cold_start_appliance(self, process_type: str) -> str:
        return {
            "open": "door",
            "close": "door",
            "unlock": "key",
            "eat": "food",
        }.get(process_type, "")

    def get_system_prompt(self) -> str:
        return (
            "You are an agent solving TextWorld text games. Follow the objective, "
            "move between rooms, inspect objects, manage inventory, and execute the "
            "available command that advances the quest."
        )

    def get_few_shot_key(self, task_type: str) -> str:
        if task_type.startswith("textworld_cooking"):
            return "textworld_cooking"
        if task_type.startswith("textworld_treasure"):
            return "textworld_treasure"
        return "textworld"

    def get_valid_commands(self) -> tuple[str, ...]:
        return (
            "look", "inventory", "go ", "north", "south", "east", "west", "up", "down",
            "take ", "get ", "pick up ", "drop ", "put ", "insert ", "open ", "close ",
            "unlock ", "lock ", "eat ", "read ", "examine ", "prepare ", "cook ",
            "slice ", "dice ", "chop ",
        )

    def get_action_list_text(self) -> str:
        return (
            "Use exactly one available TextWorld command. Common commands include "
            "look, inventory, go north/south/east/west, take <object>, drop <object>, "
            "open <object>, close <object>, unlock <object> with <key>, examine <object>, "
            "prepare meal, eat meal, cook <food> with <appliance>, and slice/dice/chop "
            "<food> with <knife>. Use examine cookbook to read recipes. When an available "
            "action list is shown, choose one action from that list exactly. If the action "
            "you wanted is not listed, do not invent it; choose the listed action that best "
            "advances the same objective."
        )

    def format_trajectory_for_reflection(self, trajectory: list[dict]) -> str:
        lines = []
        for entry in trajectory[-15:]:
            action = entry.get("action", "")
            obs = re.sub(r"\s+", " ", entry.get("observation", "") or "").strip()
            if action:
                lines.append(f"- {action} -> {obs[:180]}")
        return "\n".join(lines) if lines else "(empty)"

    def get_recep_from_action(self, action: str) -> str:
        return self.action_matches_goto(action) or ""

    @staticmethod
    def _clean_name(name: str) -> str:
        return re.sub(r"\s+", " ", name).strip(" .,:;").lower()

    @staticmethod
    def _fallback_type(name: str) -> str:
        parts = re.findall(r"[a-z0-9]+", (name or "").lower())
        return "".join(part.capitalize() for part in parts) + "Type" if parts else ""

    @staticmethod
    def _looks_like_object(name: str) -> bool:
        if not name:
            return False
        if len(name.split()) > 5:
            return False
        noisy = (
            "textworld", "virtual machine", "how to play", "first thing",
            "attempt to", "within the", "round of", "thing i need",
        )
        return not any(token in name for token in noisy)

    @staticmethod
    def _extract_inline_objective(obs: str) -> str:
        text = re.sub(r"\s+", " ", obs or "").strip()
        for pattern in (
            r"(Explore the map,\s*find\s+.+?\s+finish the game\.)",
            r"(You are hungry!\s*Let's cook a delicious meal\..+?enjoy your meal!)",
        ):
            match = re.search(pattern, text, flags=re.I)
            if match:
                return match.group(1).strip()
        return ""

    @staticmethod
    def _match_candidate(action: str, candidates: list[str]) -> str:
        action_low = action.lower().strip()
        candidates_lower = {c.lower().strip(): c for c in candidates}
        if action_low in candidates_lower:
            return candidates_lower[action_low]

        action_canon = TextWorldAdapter._canonical_action(action_low)
        for c_low, candidate in candidates_lower.items():
            if TextWorldAdapter._canonical_action(c_low) == action_canon:
                return candidate

        for c_low, candidate in candidates_lower.items():
            if action_low and (action_low in c_low or c_low in action_low):
                return candidate
        return ""

    @staticmethod
    def _canonical_action(action: str) -> str:
        text = re.sub(r"\b(the|a|an)\b", " ", action.lower())
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    @staticmethod
    def _object_from_action(action: str, verb: str) -> str:
        rest = action[len(verb):].strip()
        rest = re.split(r"\s+(?:with|from|in|on|into)\s+", rest, maxsplit=1)[0]
        return rest.strip()

    @staticmethod
    def _extract_target(desc: str) -> str:
        if not desc:
            return ""
        quoted = re.findall(r'"([^"]+)"', desc)
        if quoted:
            return quoted[-1].lower()
        for pattern in (
            r"(?:take|get|pick up|eat|drop|open|unlock)\s+(?:the\s+|a\s+|an\s+)?([a-z][a-z0-9 -]+)",
            r"(?:retrieve|find)\s+(?:the\s+|a\s+|an\s+)?([a-z][a-z0-9 -]+)",
        ):
            match = re.search(pattern, desc.lower())
            if match:
                return match.group(1).strip(" .,:;")
        return ""

    @staticmethod
    def _extract_processing(desc: str) -> str:
        low = desc.lower()
        for verb in ("eat", "unlock", "open", "take", "drop", "put"):
            if verb in low:
                return verb
        return ""
