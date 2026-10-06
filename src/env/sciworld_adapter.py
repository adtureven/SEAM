"""
ScienceWorld adapter for Seam.

The adapter uses broad symbolic parsing so the causal-memory stack can operate
on ScienceWorld-like text without hard-coding a specific task family.
"""

import re
from env.adapter import EnvironmentAdapter


COMMON_OBJECT_TYPES = {
    "water": "WaterType",
    "salt": "SaltType",
    "sugar": "SugarType",
    "soil": "SoilType",
    "sand": "SandType",
    "plant": "PlantType",
    "seed": "SeedType",
    "thermometer": "ThermometerType",
    "beaker": "BeakerType",
    "cup": "CupType",
    "mug": "MugType",
    "glass": "GlassType",
    "bottle": "BottleType",
    "jar": "JarType",
    "box": "BoxType",
    "battery": "BatteryType",
    "wire": "WireType",
    "light bulb": "LightBulbType",
    "bulb": "LightBulbType",
    "switch": "SwitchType",
    "magnet": "MagnetType",
    "metal": "MetalType",
    "wood": "WoodType",
    "paper": "PaperType",
    "rock": "RockType",
}

COMMON_RECEPTACLE_TYPES = {
    "room": "RoomType",
    "kitchen": "KitchenType",
    "lab": "LabType",
    "laboratory": "LabType",
    "table": "TableType",
    "counter": "CounterType",
    "desk": "DeskType",
    "shelf": "ShelfType",
    "drawer": "DrawerType",
    "cabinet": "CabinetType",
    "sink": "SinkType",
    "stove": "StoveType",
    "burner": "BurnerType",
    "freezer": "FreezerType",
    "fridge": "FridgeType",
    "container": "ContainerType",
}

OBJECT_CATEGORIES = {
    "liquid": {"water"},
    "material": {"salt", "sugar", "soil", "sand", "metal", "wood", "paper", "rock"},
    "organism": {"plant", "seed"},
    "tool": {"thermometer", "magnet"},
    "container": {"beaker", "cup", "mug", "glass", "bottle", "jar", "box"},
    "circuit": {"battery", "wire", "light bulb", "bulb", "switch"},
}

RECEPTACLE_CATEGORIES = {
    "workspace": {"table", "counter", "desk", "shelf"},
    "storage": {"drawer", "cabinet", "container"},
    "room": {"room", "kitchen", "lab", "laboratory"},
    "appliance": {"sink", "stove", "burner", "freezer", "fridge"},
}

_obj_to_category = {item: cat for cat, items in OBJECT_CATEGORIES.items() for item in items}
_recep_to_category = {item: cat for cat, items in RECEPTACLE_CATEGORIES.items() for item in items}


class SciWorldAdapter(EnvironmentAdapter):
    def parse_observation(self, obs: str) -> dict:
        low = obs.lower()
        result: dict = {}

        loc = re.search(r"(?:you are in|you are at|location:)\s+(?:the\s+)?([\w\s]+?)(?:\.|\n|$)", low)
        if loc:
            result["arrived_at"] = loc.group(1).strip().split(",")[0]

        objects = []
        for name in sorted(COMMON_OBJECT_TYPES, key=len, reverse=True):
            if re.search(rf"\b{name}\b", low):
                objects.append(name)
        if objects:
            result["objects_here"] = objects

        if any(x in low for x in ("you pick up", "you take", "taken.")):
            result["picked_up"] = True
        if any(x in low for x in ("you put", "you drop", "placed", "moved")):
            result["put_down"] = True
        if any(x in low for x in ("heated", "warmer", "temperature increases")):
            result["processed"] = "heat"
        elif any(x in low for x in ("cooled", "colder", "temperature decreases")):
            result["processed"] = "cool"
        elif any(x in low for x in ("mixed", "dissolved", "combined")):
            result["processed"] = "mix"
        if any(x in low for x in ("turned on", "activated", "use the")):
            result["used_device"] = True
        if any(x in low for x in ("can't", "cannot", "nothing happens", "not possible", "invalid")):
            result["error"] = obs
        return result

    def obj_name_to_type(self, name: str) -> str:
        n = re.sub(r"\s+\d+$", "", name.lower().strip())
        if n in COMMON_OBJECT_TYPES:
            return COMMON_OBJECT_TYPES[n]
        for key, val in COMMON_OBJECT_TYPES.items():
            if n == key.split()[-1]:
                return val
        return self._fallback_type(n)

    def recep_name_to_type(self, name: str) -> str:
        n = re.sub(r"\s+\d+$", "", name.lower().strip()).split()[0]
        return COMMON_RECEPTACLE_TYPES.get(n, self._fallback_type(n))

    def parse_task(self, obs: str, task_type: str = "") -> dict:
        desc = ""
        m = re.search(r"(?:your task is to:|task:|goal:)\s*(.+)", obs, re.I)
        if m:
            desc = m.group(1).lower()
        target = ""
        for name in sorted(COMMON_OBJECT_TYPES, key=len, reverse=True):
            if name in desc:
                target = name
                break
        processing = ""
        for proc in ("heat", "cool", "mix", "measure", "activate"):
            if proc in desc:
                processing = proc
                break
        return {
            "task_type": task_type or "scienceworld_task",
            "target_obj": target,
            "target_obj_type": self.obj_name_to_type(target) if target else "",
            "target_recep": "",
            "processing": processing,
        }

    def action_matches_goto(self, action: str) -> str | None:
        a = action.lower().strip()
        if a.startswith("go to "):
            return a[6:].strip()
        if a.startswith("move to "):
            return a[8:].strip()
        if a.startswith("go "):
            return a[3:].strip()
        return None

    def action_matches_take(self, action: str, target_obj: str) -> bool:
        a = action.lower().strip()
        return a.startswith(("take ", "pick up ")) and (not target_obj or target_obj in a)

    def action_matches_process(self, action: str, process_type: str) -> bool:
        return action.lower().strip().startswith((process_type + " ", "use "))

    def action_matches_put(self, action: str) -> bool:
        return action.lower().strip().startswith(("put ", "place ", "drop "))

    def action_matches_use_device(self, action: str) -> bool:
        return action.lower().strip().startswith(("use ", "activate ", "turn on "))

    def extract_action_effect(self, action: str) -> tuple[str, str] | None:
        a = action.lower().strip()
        for proc in ("heat", "cool", "mix", "measure"):
            if a.startswith(proc):
                if " with " in a:
                    return (proc, a.split(" with ")[-1].split()[0])
                return (proc, "")
        if a.startswith(("use ", "activate ", "turn on ")):
            return ("activate", a.split()[-1])
        return None

    def infer_appliance_type(self, appliance_name: str) -> str:
        n = re.sub(r"\s+\d+$", "", appliance_name.lower().strip())
        return COMMON_RECEPTACLE_TYPES.get(n) or COMMON_OBJECT_TYPES.get(n, self._fallback_type(n))

    def get_object_category(self, obj_type: str) -> str | None:
        base = obj_type.replace("Type", "").lower()
        for key, cat in _obj_to_category.items():
            if key.replace(" ", "") == base:
                return cat
        return None

    def get_receptacle_category(self, recep_type: str) -> str | None:
        base = recep_type.replace("Type", "").lower()
        return _recep_to_category.get(base)

    def get_cold_start_appliance(self, process_type: str) -> str:
        return {
            "heat": "stove",
            "cool": "freezer",
            "mix": "container",
            "measure": "thermometer",
            "activate": "switch",
        }.get(process_type, "")

    def get_system_prompt(self) -> str:
        return (
            "You are an agent solving ScienceWorld experiment tasks in a text "
            "environment. Read the task objective literally, identify the target "
            "object, instrument, room, and required final answer object, then make "
            "one environment action at a time. When an available-action list is "
            "shown, copy exactly one action from that list. Do not invent action "
            "syntax, object names, or observations."
        )

    def get_few_shot_key(self, task_type: str) -> str:
        t = (task_type or "").lower()
        if t in {"boil", "freeze", "melt", "change-the-state-of-matter-of"}:
            return "sciworld_matter"
        if t.startswith("measure-melting-point") or t == "use-thermometer":
            return "sciworld_measure"
        if t.startswith("test-conductivity") or t.startswith("power-component"):
            return "sciworld_elec"
        if (
            t.startswith("find-")
            or t.startswith("grow-")
            or t.startswith("lifespan-")
            or t.startswith("mendelian-genetics")
            or t.startswith("identify-life-stages")
        ):
            return "sciworld_bio"
        if t.startswith("chemistry-") or t.startswith("inclined-plane"):
            return "sciworld_chemphys"
        return "sciworld"

    def get_valid_commands(self) -> tuple[str, ...]:
        return ("look", "inventory", "go", "go to", "move to", "open", "close",
                "take", "pick up", "put", "place", "drop", "use", "activate",
                "turn on", "heat", "cool", "mix", "measure", "examine",
                "focus", "focus on", "dunk", "read", "eat")

    def get_action_list_text(self) -> str:
        return (
            "Valid actions include look, inventory, go/go to/move to, open, close, "
            "take/pick up, put/place/drop, use/activate/turn on, heat, cool, mix, "
            "measure, examine, focus/focus on, dunk, read, eat. In ScienceWorld, "
            "the exact wording matters: if available actions are provided, choose "
            "one listed action verbatim. Prefer task-relevant actions: move to the "
            "named room, focus on the named object/instrument, perform the required "
            "experiment, then focus on or place the required answer object."
        )

    def format_trajectory_for_reflection(self, trajectory: list[dict]) -> str:
        parts = []
        for entry in trajectory[-15:]:
            act = entry.get("action", "")
            obs = (entry.get("observation") or "").lower()
            if act:
                result = "FAILED" if any(x in obs for x in ("can't", "cannot", "nothing happens")) else "OK"
                parts.append(f"- {act} -> {result}")
        return "\n".join(parts) if parts else "(empty)"

    def get_recep_from_action(self, action: str) -> str:
        return self.action_matches_goto(action) or ""

    def _fallback_type(self, name: str) -> str:
        if not name:
            return ""
        parts = re.findall(r"[a-z0-9]+", name)
        return "".join(p.capitalize() for p in parts) + "Type" if parts else ""
