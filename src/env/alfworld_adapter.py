import re
from env.adapter import EnvironmentAdapter


# ─── ALFWorld Type Mappings ───

OBJ_TYPE_MAP = {
    "apple": "AppleType", "book": "BookType", "bowl": "BowlType",
    "bread": "BreadType", "butterknife": "ButterKnifeType",
    "candle": "CandleType", "cd": "CDType", "cellphone": "CellPhoneType",
    "cloth": "ClothType", "creditcard": "CreditCardType",
    "cup": "CupType", "dishsponge": "DishSpongeType",
    "egg": "EggType", "fork": "ForkType", "glassbottle": "GlassBottleType",
    "handtowel": "HandTowelType", "kettle": "KettleType",
    "knife": "KnifeType", "ladle": "LadleType", "laptop": "LaptopType",
    "lettuce": "LettuceType", "mug": "MugType", "newspaper": "NewspaperType",
    "pan": "PanType", "papertowelroll": "PaperTowelRollType",
    "pen": "PenType", "pencil": "PencilType", "peppershaker": "PepperShakerType",
    "pillow": "PillowType", "plate": "PlateType", "plunger": "PlungerType",
    "pot": "PotType", "potato": "PotatoType", "remote": "RemoteControlType",
    "remotecontrol": "RemoteControlType",
    "saltshaker": "SaltShakerType", "soapbar": "SoapBarType",
    "soapbottle": "SoapBottleType", "spatula": "SpatulaType",
    "spoon": "SpoonType", "spraybottle": "SprayBottleType",
    "statue": "StatueType", "stoveknob": "StoveKnobType",
    "tabletopdecor": "TableTopDecorType", "tissuebox": "TissueBoxType",
    "toiletpaper": "ToiletPaperType", "tomato": "TomatoType",
    "towel": "TowelType", "vase": "VaseType", "watch": "WatchType",
    "wateringcan": "WateringCanType", "winebottle": "WineBottleType",
    "alarmclock": "AlarmClockType", "baseballbat": "BaseballBatType",
    "basketball": "BasketBallType", "bathtowel": "BathTowelType",
    "boots": "BootsType", "box": "BoxType", "desklamp": "DeskLampType",
    "floorlamp": "FloorLampType", "houseplant": "HousePlantType",
    "keychain": "KeyChainType", "lightswitch": "LightSwitchType",
    "mirror": "MirrorType", "scrubbrush": "ScrubBrushType",
    "tennisracket": "TennisRacketType", "teddybear": "TeddyBearType",
}

RECEP_TYPE_MAP = {
    "countertop": "CounterTopType", "cabinet": "CabinetType",
    "drawer": "DrawerType", "fridge": "FridgeType",
    "microwave": "MicrowaveType", "shelf": "ShelfType",
    "diningtable": "DiningTableType", "sidetable": "SideTableType",
    "coffeetable": "CoffeeTableType", "desk": "DeskType",
    "dresser": "DresserType", "garbagecan": "GarbageCanType",
    "sink": "SinkType", "sinkbasin": "SinkBasinType",
    "bathtub": "BathtubType", "bathtubbasin": "BathtubBasinType",
    "toilet": "ToiletType", "sofa": "SofaType",
    "armchair": "ArmChairType", "bed": "BedType",
    "ottoman": "OttomanType", "stoveburner": "StoveBurnerType",
    "toaster": "ToasterType", "tvstand": "TVStandType",
    "safe": "SafeType", "cart": "CartType",
    "table": "DiningTableType",
    "coffeemachine": "CoffeeMachineType",
    "laundryhamper": "LaundryHamperType",
}

TASK_PATTERNS = {
    "heat": re.compile(r"(?:put|place)\s+a\s+hot\s+(\w+)", re.I),
    "cool": re.compile(r"(?:put|place)\s+a\s+(?:cool|cold)\s+(\w+)", re.I),
    "clean": re.compile(r"(?:put|place)\s+a\s+clean\s+(\w+)", re.I),
    "light": re.compile(r"(?:examine|look at)\s+(?:the\s+|a\s+)?(\w+)\s+(?:under|with|in)\s+(?:the\s+)?(?:desk\s*)?lamp", re.I),
    "place": re.compile(r"(?:put|place)\s+(?:a\s+|some\s+)?(\w+)\s+(?:in|on)\s+(\w+)", re.I),
    "two": re.compile(r"(?:put|find)\s+(?:two|2)\s+(\w+)", re.I),
    "process_and": re.compile(r"(?:heat|cool|clean)\s+some\s+(\w+)\s+and\s+put", re.I),
}

TASK_TO_REACT_KEY = {
    "pick_and_place_simple": "put",
    "pick_clean_then_place_in_recep": "clean",
    "pick_heat_then_place_in_recep": "heat",
    "pick_cool_then_place_in_recep": "cool",
    "pick_two_obj_and_place": "puttwo",
    "look_at_obj_in_light": "examine",
}

OBJECT_CATEGORIES = {
    "food": {"egg", "apple", "bread", "potato", "tomato", "lettuce"},
    "kitchenware": {"cup", "mug", "bowl", "plate", "pan", "pot", "kettle",
                    "fork", "knife", "spoon", "spatula", "ladle", "butterknife"},
    "cleaning": {"cloth", "dishsponge", "soapbar", "soapbottle", "spraybottle",
                 "scrubbrush", "plunger"},
    "lighting": {"candle", "desklamp", "floorlamp", "lightswitch"},
    "personal": {"cellphone", "creditcard", "keychain", "watch", "pen", "pencil",
                 "book", "newspaper", "cd", "laptop", "remote", "remotecontrol",
                 "alarmclock"},
    "decoration": {"statue", "vase", "houseplant", "pillow", "teddybear",
                   "tabletopdecor", "winebottle", "glassbottle"},
    "hygiene": {"handtowel", "towel", "bathtowel", "toiletpaper",
                "papertowelroll", "tissuebox"},
    "sports": {"baseballbat", "basketball", "tennisracket", "boots"},
    "container_item": {"box", "wateringcan", "peppershaker", "saltshaker"},
}

RECEPTACLE_CATEGORIES = {
    "kitchen_surface": {"countertop", "diningtable", "stoveburner", "toaster"},
    "kitchen_storage": {"cabinet", "drawer", "fridge", "microwave"},
    "bathroom": {"sinkbasin", "sink", "bathtub", "bathtubbasin", "toilet"},
    "living_surface": {"sidetable", "coffeetable", "desk", "dresser",
                       "tvstand", "ottoman", "shelf"},
    "seating": {"sofa", "armchair", "bed"},
    "waste": {"garbagecan"},
    "secure": {"safe"},
    "mobile": {"cart"},
}

COLD_START_APPLIANCE = {
    "heat": "microwave",
    "cool": "fridge",
    "clean": "sinkbasin",
    "use_lamp": "desklamp",
}

APPLIANCE_TYPE_MAP = {
    "microwave": "MicrowaveType",
    "fridge": "FridgeType",
    "sinkbasin": "SinkBasinType",
    "sink": "SinkBasinType",
    "desklamp": "DeskLampType",
    "floorlamp": "FloorLampType",
}

# Build reverse category lookups
_obj_to_category: dict[str, str] = {}
for _cat, _items in OBJECT_CATEGORIES.items():
    for _item in _items:
        _obj_to_category[_item] = _cat

_recep_to_category: dict[str, str] = {}
for _cat, _items in RECEPTACLE_CATEGORIES.items():
    for _item in _items:
        _recep_to_category[_item] = _cat


class ALFWorldAdapter(EnvironmentAdapter):

    def parse_observation(self, obs: str) -> dict:
        result: dict = {}
        low = obs.lower()

        if "you arrive" in low or "you are facing" in low:
            m = re.search(r"(?:arrive at|facing the)\s+([\w\s]+?)(?:\.|,)", low)
            if m:
                raw = m.group(1).strip()
                parts = raw.split()
                name = f"{parts[0]} {parts[1]}" if len(parts) >= 2 and parts[1].isdigit() else parts[0]
                result["arrived_at"] = name
            obj_pattern = re.compile(r"(?:a|an)\s+([\w\s]+?)(?:\s+\d+)?(?:,|\.|$)")
            found = []
            for m in obj_pattern.finditer(obs):
                obj_text = m.group(1).strip().split()[0].lower()
                if obj_text in OBJ_TYPE_MAP:
                    found.append(obj_text)
            result["objects_here"] = found

        if "you pick up" in low:
            result["picked_up"] = True
        if "you put" in low:
            result["put_down"] = True
        if "you heat" in low:
            result["processed"] = "heat"
        elif "you cool" in low:
            result["processed"] = "cool"
        elif "you clean" in low:
            result["processed"] = "clean"
        if "you turn on" in low or "you use" in low:
            result["used_device"] = True
        if "nothing happens" in low or "can't" in low or "not holding" in low:
            result["error"] = obs
        if "closed" in low and "you need to open" in low:
            result["error"] = obs
        return result

    def obj_name_to_type(self, name: str) -> str:
        base = re.sub(r"\s+\d+$", "", name.lower().strip())
        return OBJ_TYPE_MAP.get(base, "")

    def recep_name_to_type(self, name: str) -> str:
        base = re.sub(r"\s+\d+$", "", name.lower().strip()).split()[0]
        return RECEP_TYPE_MAP.get(base, "")

    def parse_task(self, obs: str, task_type: str = "") -> dict:
        desc_match = re.search(r"Your task is to: (.+)", obs)
        if not desc_match:
            return {"task_type": task_type or "", "target_obj": "", "target_obj_type": "",
                    "target_recep": "", "processing": ""}
        desc = desc_match.group(1).lower()

        tt = task_type
        processing = ""
        if tt == "pick_heat_then_place_in_recep" or (not tt and TASK_PATTERNS["heat"].search(desc)):
            tt = "pick_heat_then_place_in_recep"
            processing = "heat"
        elif tt == "pick_cool_then_place_in_recep" or (not tt and TASK_PATTERNS["cool"].search(desc)):
            tt = "pick_cool_then_place_in_recep"
            processing = "cool"
        elif tt == "pick_clean_then_place_in_recep" or (not tt and TASK_PATTERNS["clean"].search(desc)):
            tt = "pick_clean_then_place_in_recep"
            processing = "clean"
        elif tt == "look_at_obj_in_light" or (not tt and TASK_PATTERNS["light"].search(desc)):
            tt = "look_at_obj_in_light"
            processing = "use_lamp"
        elif tt == "pick_two_obj_and_place" or (not tt and TASK_PATTERNS["two"].search(desc)):
            tt = "pick_two_obj_and_place"
        else:
            tt = tt or "pick_and_place_simple"

        target_obj = self._extract_target_obj(desc)
        target_obj_type = OBJ_TYPE_MAP.get(target_obj.rstrip("s"), "")

        target_recep = ""
        recep_matches = re.findall(r"\b(?:in|on)\s+(?:a\s+|the\s+)?(\w+)\b", desc)
        if recep_matches:
            for candidate in reversed(recep_matches):
                if candidate not in ("a", "the", "lamp", "light", "it", "them"):
                    target_recep = candidate
                    break

        return {
            "task_type": tt,
            "target_obj": target_obj,
            "target_obj_type": target_obj_type,
            "target_recep": target_recep,
            "processing": processing,
        }

    def _extract_target_obj(self, desc: str) -> str:
        m = TASK_PATTERNS["process_and"].search(desc)
        if m:
            return m.group(1)
        for key in ("light", "two", "heat", "cool", "clean", "place"):
            m = TASK_PATTERNS[key].search(desc)
            if m:
                obj = m.group(1)
                if obj not in ("it", "them", "the", "a", "some"):
                    return obj
        obj_words = re.findall(
            r"\b(" + "|".join(sorted(OBJ_TYPE_MAP.keys(), key=len, reverse=True)) + r")\b",
            desc
        )
        return obj_words[0] if obj_words else ""

    def action_matches_goto(self, action: str) -> str | None:
        a = action.lower().strip()
        if a.startswith("go to "):
            return a[6:].strip()
        return None

    def action_matches_take(self, action: str, target_obj: str) -> bool:
        a = action.lower().strip()
        return a.startswith("take ") and target_obj and target_obj in a.split()

    def action_matches_process(self, action: str, process_type: str) -> bool:
        a = action.lower().strip()
        if process_type == "use_lamp":
            return a.startswith("use ")
        return a.startswith(f"{process_type} ")

    def action_matches_put(self, action: str) -> bool:
        a = action.lower().strip()
        return a.startswith("put ") or a.startswith("move ")

    def action_matches_use_device(self, action: str) -> bool:
        return action.lower().strip().startswith("use ")

    def extract_action_effect(self, action: str) -> tuple[str, str] | None:
        a = action.lower().strip()
        for proc in ("heat", "cool", "clean"):
            if a.startswith(f"{proc} ") and "with" in a:
                appliance = a.split("with")[-1].strip().split()[0]
                return (proc, appliance)
        if a.startswith("use ") and "lamp" in a:
            return ("use_lamp", "desklamp")
        return None

    def infer_appliance_type(self, appliance_name: str) -> str:
        base = re.sub(r"\s+\d+$", "", appliance_name.lower().strip())
        return APPLIANCE_TYPE_MAP.get(base, "")

    def get_object_category(self, obj_type: str) -> str | None:
        base = obj_type.replace("Type", "").lower()
        return _obj_to_category.get(base)

    def get_receptacle_category(self, recep_type: str) -> str | None:
        base = recep_type.replace("Type", "").lower()
        return _recep_to_category.get(base)

    def get_cold_start_appliance(self, process_type: str) -> str:
        return COLD_START_APPLIANCE.get(process_type, "")

    def get_system_prompt(self) -> str:
        return "Interact with a household to solve a task. Here are two examples."

    def get_few_shot_key(self, task_type: str) -> str:
        return TASK_TO_REACT_KEY.get(task_type, "put")

    def get_valid_commands(self) -> tuple[str, ...]:
        return ("look", "inventory", "go to", "open", "close", "take",
                "put", "move", "heat", "clean", "cool", "use", "examine")

    def get_action_list_text(self) -> str:
        return "Valid actions: look, go to, open, close, take, put, heat, cool, clean, use, examine, inventory."

    def format_trajectory_for_reflection(self, trajectory: list[dict]) -> str:
        parts = []
        for entry in trajectory[-15:]:
            act = entry.get("action", "")
            obs = (entry.get("observation") or "").lower()
            if not act:
                continue
            if act.startswith("go to "):
                recep = act[6:].strip()
                objs = re.findall(r"(?:a|an)\s+([\w]+)", obs)
                objs_str = ", ".join(objs[:4]) if objs else "nothing"
                parts.append(f"- went to {recep}, found {objs_str}")
            elif act.startswith("take "):
                result = "OK" if "you pick up" in obs else "FAILED"
                parts.append(f"- {act} → {result}")
            elif act.startswith("put "):
                result = "OK" if "you put" in obs else "FAILED"
                parts.append(f"- {act} → {result}")
            elif act.startswith(("heat ", "cool ", "clean ")):
                if "nothing happens" in obs or "not holding" in obs or "can't" in obs:
                    parts.append(f"- {act} → FAILED")
                else:
                    parts.append(f"- {act} → OK")
            elif act.startswith(("use ", "open ")):
                parts.append(f"- {act}")
        return "\n".join(parts) if parts else "(empty)"

    def get_recep_from_action(self, action: str) -> str:
        a = action.lower().strip()
        if a.startswith("go to "):
            return a[6:].strip()
        return ""
