"""
Lightweight ALFWorld environment based on PDDL game files.
Parses game.tw-pddl, maintains state as a set of predicates,
executes actions by checking preconditions and applying effects,
and generates text observations.
"""

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class GameTask:
    task_name: str
    task_desc: str
    walkthrough: list[str]
    task_type: str
    game_file: str


class ALFWorldEnv:
    TASK_TYPES = {
        "pick_and_place_simple",
        "look_at_obj_in_light",
        "pick_clean_then_place_in_recep",
        "pick_heat_then_place_in_recep",
        "pick_cool_then_place_in_recep",
        "pick_two_obj_and_place",
        "pick_and_place_with_movable_recep",
    }

    def __init__(self, data_path: str, split: str = "valid_seen", max_steps: int = 30):
        self.data_path = Path(os.path.expanduser(data_path))
        self.split = split
        self.max_steps = max_steps
        self.tasks = self._load_tasks()
        self._state: set[str] = set()
        self._objects: dict[str, str] = {}
        self._receptacles: dict[str, str] = {}
        self._object_names: dict[str, str] = {}
        self._game_data: dict = {}
        self._step_count = 0
        self._done = False
        self._current_task: GameTask | None = None

    def _load_tasks(self) -> list[GameTask]:
        split_dir = self.data_path / self.split
        tasks = []
        for task_dir in sorted(split_dir.iterdir()):
            if not task_dir.is_dir():
                continue
            task_type = task_dir.name.split("-")[0]
            for trial_dir in task_dir.iterdir():
                game_file = trial_dir / "game.tw-pddl"
                traj_file = trial_dir / "traj_data.json"
                if not game_file.exists():
                    continue
                with open(game_file) as f:
                    game_data = json.load(f)
                if not game_data.get("solvable", False):
                    continue
                task_desc = self._extract_task_from_grammar(game_data.get("grammar", ""))
                tasks.append(GameTask(
                    task_name=task_dir.name,
                    task_desc=task_desc,
                    walkthrough=game_data.get("walkthrough", []),
                    task_type=task_type,
                    game_file=str(game_file),
                ))
        return tasks

    def _extract_task_from_grammar(self, grammar_str: str) -> str:
        match = re.search(r'"task":\s*\[\s*\{\s*"rhs":\s*"Your task is to: ([^"]+)"', grammar_str)
        if match:
            return match.group(1)
        return "unknown task"

    def num_tasks(self) -> int:
        return len(self.tasks)

    def reset(self, task_idx: int) -> tuple[str, GameTask]:
        task = self.tasks[task_idx]
        self._current_task = task
        with open(task.game_file) as f:
            self._game_data = json.load(f)
        self._state = set()
        self._objects = {}
        self._receptacles = {}
        self._object_names = {}
        self._step_count = 0
        self._done = False
        self._parse_initial_state()
        obs = self._generate_look_observation()
        intro = f"-= Welcome to ALFWorld =-\n\n{obs}\n\nYour task is to: {task.task_desc}"
        return intro, task

    def _parse_initial_state(self):
        pddl = self._game_data.get("pddl_problem", "")
        init_match = re.search(r"\(:init\s*(.+?)\)\s*\(:goal", pddl, re.DOTALL)
        if not init_match:
            init_match = re.search(r"\(:init\s*(.+?)\)\s*$", pddl, re.DOTALL)
        if not init_match:
            return
        init_section = init_match.group(1)
        for match in re.finditer(r"\((\w+)\s+([^)]+)\)", init_section):
            pred_name = match.group(1)
            args = match.group(2).strip().split()
            if any(a.startswith("?") for a in args):
                continue
            pred_str = f"({pred_name} {' '.join(args)})"
            self._state.add(pred_str)
            if pred_name == "objectType":
                self._objects[args[0]] = args[1]
            elif pred_name == "receptacleType":
                self._receptacles[args[0]] = args[1]
        self._build_name_map()

    def _build_name_map(self):
        type_counts: dict[str, list[str]] = {}
        all_ids = sorted(list(self._receptacles.keys()) + list(self._objects.keys()))
        for obj_id in all_ids:
            base_name = self._get_base_name(obj_id)
            if base_name not in type_counts:
                type_counts[base_name] = []
            type_counts[base_name].append(obj_id)
        for base_name, ids in type_counts.items():
            count = len(ids)
            num_ids = list(range(1, count + 1))
            for obj_id in ids:
                assigned = num_ids.pop()
                if count == 1:
                    self._object_names[obj_id] = base_name
                else:
                    self._object_names[obj_id] = f"{base_name} {assigned}"

    def _get_base_name(self, pddl_id: str) -> str:
        parts = pddl_id.split("_bar_")
        base = parts[0]
        if "basin" in pddl_id.lower():
            base += "basin"
        return base.lower().strip()

    def _get_name(self, obj_id: str) -> str:
        return self._object_names.get(obj_id, obj_id)

    def step(self, action: str) -> tuple[str, float, bool, dict]:
        if self._done:
            return "Game is over.", 0.0, True, {}
        self._step_count += 1
        action = action.strip().lower()
        if action == "look":
            obs = self._generate_look_observation()
        elif action == "inventory":
            obs = self._generate_inventory()
        elif action.startswith("go to "):
            obs = self._execute_goto(action)
        elif action.startswith("open "):
            obs = self._execute_open(action)
        elif action.startswith("close "):
            obs = self._execute_close(action)
        elif action.startswith("take "):
            obs = self._execute_take(action)
        elif action.startswith("put ") or action.startswith("move "):
            obs = self._execute_put(action)
        elif action.startswith("use "):
            obs = self._execute_use(action)
        elif action.startswith("heat "):
            obs = self._execute_heat(action)
        elif action.startswith("clean "):
            obs = self._execute_clean(action)
        elif action.startswith("cool "):
            obs = self._execute_cool(action)
        elif action.startswith("examine "):
            obs = self._execute_examine(action)
        else:
            obs = "Unknown action. Try: look, go to, take, put, open, close, use, heat, clean, cool, examine."
        goal_reached = self._check_goal()
        if goal_reached:
            self._done = True
            reward = 1.0
            obs += "\nYou have completed the task!"
        elif self._step_count >= self.max_steps:
            self._done = True
            reward = 0.0
            obs += "\nYou have run out of steps."
        else:
            reward = 0.0
        info = {
            "steps": self._step_count,
            "goal_reached": goal_reached,
        }
        return obs, reward, self._done, info

    def _get_agent_location(self) -> str | None:
        for pred in self._state:
            m = re.match(r"\(atLocation agent1 (.+)\)", pred)
            if m:
                return m.group(1)
        return None

    def _get_receptacles_at_location(self, loc: str) -> list[str]:
        result = []
        for pred in self._state:
            m = re.match(r"\(receptacleAtLocation (.+) (.+)\)", pred)
            if m and m.group(2) == loc:
                result.append(m.group(1))
        return result

    def _get_objects_in_receptacle(self, recep: str) -> list[str]:
        result = []
        for pred in self._state:
            m = re.match(r"\(inReceptacle (.+) (.+)\)", pred)
            if m and m.group(2) == recep:
                result.append(m.group(1))
        return result

    def _is_holding(self, obj: str) -> bool:
        return f"(holds agent1 {obj})" in self._state

    def _get_held_object(self) -> str | None:
        for pred in self._state:
            m = re.match(r"\(holds agent1 (.+)\)", pred)
            if m:
                return m.group(1)
        return None

    def _normalize_name(self, name: str) -> str:
        return name.strip().lower().replace(" ", "")

    def _names_match(self, query: str, target: str) -> bool:
        q = self._normalize_name(query)
        t = self._normalize_name(target)
        if q == t:
            return True
        q_match = re.match(r"(.+?)(\d+)$", q)
        t_match = re.match(r"(.+?)(\d+)$", t)
        if q_match and not t_match:
            return q_match.group(1) == t
        if q_match and t_match:
            return q_match.group(1) == t_match.group(1) and q_match.group(2) == t_match.group(2)
        return q == t or q in t or t.startswith(q)

    def _find_object_by_name(self, name: str, candidates: list[str] | None = None) -> str | None:
        search_space = candidates if candidates else list(self._objects.keys()) + list(self._receptacles.keys())
        for obj_id in search_space:
            if self._names_match(name, self._get_name(obj_id)):
                return obj_id
        return None

    def _find_receptacle_by_name(self, name: str) -> str | None:
        for rec_id in self._receptacles:
            if self._names_match(name, self._get_name(rec_id)):
                return rec_id
        return None

    def _generate_look_observation(self) -> str:
        loc = self._get_agent_location()
        if not loc:
            return "You are somewhere."
        receptacles = self._get_receptacles_at_location(loc)
        if receptacles:
            recep = receptacles[0]
            recep_name = self._get_name(recep)
            is_openable = f"(openable {recep})" in self._state
            is_opened = f"(opened {recep})" in self._state
            if is_openable and is_opened:
                objects = self._get_objects_in_receptacle(recep)
                obj_names = [f"a {self._get_name(o)}" for o in objects]
                obj_str = ", ".join(obj_names) if obj_names else "nothing"
                obs = f"You are facing the {recep_name}. The {recep_name} is open. In it, you see {obj_str}."
            elif is_openable:
                obs = f"You are facing the {recep_name}. The {recep_name} is closed."
            else:
                objects = self._get_objects_in_receptacle(recep)
                obj_names = [f"a {self._get_name(o)}" for o in objects]
                obj_str = ", ".join(obj_names) if obj_names else "nothing"
                obs = f"You are facing the {recep_name}. On the {recep_name}, you see {obj_str}."
        else:
            all_receps = []
            for pred in self._state:
                m = re.match(r"\(receptacleAtLocation (.+) .+\)", pred)
                if m:
                    all_receps.append(self._get_name(m.group(1)))
            obs = f"You are in the middle of a room. Looking quickly around you, you see {', '.join(set(all_receps))}."
        return obs

    def _generate_inventory(self) -> str:
        held = self._get_held_object()
        if held:
            return f"You are carrying: a {self._get_name(held)}."
        return "You are not carrying anything."

    def _execute_goto(self, action: str) -> str:
        target_name = action[len("go to "):].strip()
        recep_id = self._find_receptacle_by_name(target_name)
        if not recep_id:
            return f"Cannot find {target_name}."
        target_loc = None
        for pred in self._state:
            m = re.match(r"\(receptacleAtLocation (.+) (.+)\)", pred)
            if m and m.group(1) == recep_id:
                target_loc = m.group(2)
                break
        if not target_loc:
            return f"Cannot find location of {target_name}."
        current_loc = self._get_agent_location()
        if current_loc:
            self._state.discard(f"(atLocation agent1 {current_loc})")
        self._state.add(f"(atLocation agent1 {target_loc})")
        recep_name = self._get_name(recep_id)
        is_openable = f"(openable {recep_id})" in self._state
        is_opened = f"(opened {recep_id})" in self._state
        if is_openable and is_opened:
            objects = self._get_objects_in_receptacle(recep_id)
            obj_names = [f"a {self._get_name(o)}" for o in objects]
            obj_str = ", ".join(obj_names) if obj_names else "nothing"
            return f"You arrive at {recep_name}. The {recep_name} is open. In it, you see {obj_str}."
        elif is_openable:
            return f"You arrive at {recep_name}. The {recep_name} is closed."
        else:
            objects = self._get_objects_in_receptacle(recep_id)
            obj_names = [f"a {self._get_name(o)}" for o in objects]
            obj_str = ", ".join(obj_names) if obj_names else "nothing"
            return f"You arrive at {recep_name}. On the {recep_name}, you see {obj_str}."

    def _execute_open(self, action: str) -> str:
        target_name = action[len("open "):].strip()
        recep_id = self._find_receptacle_by_name(target_name)
        if not recep_id:
            return f"Cannot find {target_name}."
        if f"(openable {recep_id})" not in self._state:
            return f"The {self._get_name(recep_id)} is not openable."
        if f"(opened {recep_id})" in self._state:
            return f"The {self._get_name(recep_id)} is already open."
        loc = self._get_agent_location()
        for pred in self._state:
            m = re.match(r"\(receptacleAtLocation (.+) (.+)\)", pred)
            if m and m.group(1) == recep_id and m.group(2) == loc:
                self._state.add(f"(opened {recep_id})")
                objects = self._get_objects_in_receptacle(recep_id)
                obj_names = [f"a {self._get_name(o)}" for o in objects]
                obj_str = ", ".join(obj_names) if obj_names else "nothing"
                return f"You open the {self._get_name(recep_id)}. The {self._get_name(recep_id)} is open. In it, you see {obj_str}."
        return f"You need to go to the {self._get_name(recep_id)} first."

    def _execute_close(self, action: str) -> str:
        target_name = action[len("close "):].strip()
        recep_id = self._find_receptacle_by_name(target_name)
        if not recep_id:
            return f"Cannot find {target_name}."
        if f"(opened {recep_id})" not in self._state:
            return f"The {self._get_name(recep_id)} is already closed."
        self._state.discard(f"(opened {recep_id})")
        return f"You close the {self._get_name(recep_id)}."

    def _execute_take(self, action: str) -> str:
        match = re.match(r"take (.+) from (.+)", action)
        if not match:
            return "Invalid take command. Use: take <object> from <receptacle>"
        obj_name = match.group(1).strip()
        recep_name = match.group(2).strip()
        if self._get_held_object():
            return "You are already holding something."
        recep_id = self._find_receptacle_by_name(recep_name)
        if not recep_id:
            return f"Cannot find {recep_name}."
        objects_in_recep = self._get_objects_in_receptacle(recep_id)
        obj_id = self._find_object_by_name(obj_name, objects_in_recep)
        if not obj_id:
            return f"Cannot find {obj_name} in {recep_name}."
        if f"(openable {recep_id})" in self._state and f"(opened {recep_id})" not in self._state:
            return f"The {self._get_name(recep_id)} is closed. You need to open it first."
        self._state.discard(f"(inReceptacle {obj_id} {recep_id})")
        loc = self._get_agent_location()
        self._state.discard(f"(objectAtLocation {obj_id} {loc})")
        self._state.add(f"(holds agent1 {obj_id})")
        self._state.add("(holdsAny agent1)")
        return f"You pick up the {self._get_name(obj_id)} from the {self._get_name(recep_id)}."

    def _execute_put(self, action: str) -> str:
        match = re.match(r"(?:put|move) (.+) (?:to|in/on|in|on) (.+)", action)
        if not match:
            return "Invalid put command. Use: put/move <object> to/in <receptacle>"
        recep_name = match.group(2).strip()
        held = self._get_held_object()
        if not held:
            return "You are not holding anything."
        recep_id = self._find_receptacle_by_name(recep_name)
        if not recep_id:
            return f"Cannot find {recep_name}."
        if f"(openable {recep_id})" in self._state and f"(opened {recep_id})" not in self._state:
            return f"The {self._get_name(recep_id)} is closed. You need to open it first."
        self._state.discard(f"(holds agent1 {held})")
        self._state.discard("(holdsAny agent1)")
        self._state.add(f"(inReceptacle {held} {recep_id})")
        loc = self._get_agent_location()
        self._state.add(f"(objectAtLocation {held} {loc})")
        return f"You put the {self._get_name(held)} in/on the {self._get_name(recep_id)}."

    def _execute_use(self, action: str) -> str:
        target_name = action[len("use "):].strip()
        obj_id = self._find_object_by_name(target_name, list(self._objects.keys()))
        if not obj_id:
            return f"Cannot find {target_name}."
        if f"(toggleable {obj_id})" not in self._state:
            return f"You can't use the {self._get_name(obj_id)}."
        if f"(isOn {obj_id})" in self._state:
            self._state.discard(f"(isOn {obj_id})")
            self._state.add(f"(isToggled {obj_id})")
            return f"You turn off the {self._get_name(obj_id)}."
        else:
            self._state.add(f"(isOn {obj_id})")
            self._state.add(f"(isToggled {obj_id})")
            return f"You turn on the {self._get_name(obj_id)}."

    def _execute_heat(self, action: str) -> str:
        match = re.match(r"heat (.+) with (.+)", action)
        if not match:
            return "Invalid heat command. Use: heat <object> with <receptacle>"
        recep_name = match.group(2).strip()
        held = self._get_held_object()
        if not held:
            return "You are not holding anything."
        if f"(heatable {held})" not in self._state:
            return f"The {self._get_name(held)} cannot be heated."
        recep_id = self._find_receptacle_by_name(recep_name)
        if not recep_id:
            return f"Cannot find {recep_name}."
        recep_type = self._receptacles.get(recep_id, "")
        if recep_type != "MicrowaveType":
            return f"You can't heat things with the {self._get_name(recep_id)}."
        self._state.add(f"(isHot {held})")
        self._state.discard(f"(isCool {held})")
        return f"You heat the {self._get_name(held)} using the {self._get_name(recep_id)}."

    def _execute_clean(self, action: str) -> str:
        match = re.match(r"clean (.+) with (.+)", action)
        if not match:
            return "Invalid clean command. Use: clean <object> with <receptacle>"
        recep_name = match.group(2).strip()
        held = self._get_held_object()
        if not held:
            return "You are not holding anything."
        if f"(cleanable {held})" not in self._state:
            return f"The {self._get_name(held)} cannot be cleaned."
        recep_id = self._find_receptacle_by_name(recep_name)
        if not recep_id:
            return f"Cannot find {recep_name}."
        recep_type = self._receptacles.get(recep_id, "")
        if recep_type not in ("SinkType", "SinkBasinType"):
            return f"You can't clean things with the {self._get_name(recep_id)}."
        self._state.add(f"(isClean {held})")
        return f"You clean the {self._get_name(held)} using the {self._get_name(recep_id)}."

    def _execute_cool(self, action: str) -> str:
        match = re.match(r"cool (.+) with (.+)", action)
        if not match:
            return "Invalid cool command. Use: cool <object> with <receptacle>"
        recep_name = match.group(2).strip()
        held = self._get_held_object()
        if not held:
            return "You are not holding anything."
        if f"(coolable {held})" not in self._state:
            return f"The {self._get_name(held)} cannot be cooled."
        recep_id = self._find_receptacle_by_name(recep_name)
        if not recep_id:
            return f"Cannot find {recep_name}."
        recep_type = self._receptacles.get(recep_id, "")
        if recep_type != "FridgeType":
            return f"You can't cool things with the {self._get_name(recep_id)}."
        self._state.add(f"(isCool {held})")
        self._state.discard(f"(isHot {held})")
        return f"You cool the {self._get_name(held)} using the {self._get_name(recep_id)}."

    def _execute_examine(self, action: str) -> str:
        target_name = action[len("examine "):].strip()
        recep_id = self._find_receptacle_by_name(target_name)
        if recep_id:
            is_openable = f"(openable {recep_id})" in self._state
            is_opened = f"(opened {recep_id})" in self._state
            if is_openable and not is_opened:
                return f"The {self._get_name(recep_id)} is closed."
            objects = self._get_objects_in_receptacle(recep_id)
            obj_names = [f"a {self._get_name(o)}" for o in objects]
            obj_str = ", ".join(obj_names) if obj_names else "nothing"
            if is_openable:
                return f"The {self._get_name(recep_id)} is open. In it, you see {obj_str}."
            return f"On the {self._get_name(recep_id)}, you see {obj_str}."
        obj_id = self._find_object_by_name(target_name, list(self._objects.keys()))
        if obj_id:
            return self._describe_object(obj_id)
        return f"Cannot find {target_name}."

    def _describe_object(self, obj_id: str) -> str:
        name = self._get_name(obj_id)
        props = []
        if f"(isClean {obj_id})" in self._state:
            props.append("clean")
        if f"(isHot {obj_id})" in self._state:
            props.append("hot")
        if f"(isCool {obj_id})" in self._state:
            props.append("cold")
        if f"(isSliced {obj_id})" in self._state:
            props.append("sliced")
        if props:
            return f"This is a {' and '.join(props)} {name}."
        if f"(toggleable {obj_id})" in self._state:
            if f"(isOn {obj_id})" in self._state:
                return f"This {name} is on."
            return f"This {name} is off."
        return f"There's nothing special about {name}."

    def _check_goal(self) -> bool:
        pddl = self._game_data.get("pddl_problem", "")
        goal_match = re.search(r":goal\s*\((.+)", pddl, re.DOTALL)
        if not goal_match:
            return False
        goal_text = goal_match.group(1)
        goal_conditions = self._extract_goal_conditions(goal_text)
        return self._evaluate_goal(goal_conditions)

    def _extract_goal_conditions(self, goal_text: str) -> list[dict]:
        conditions = []
        if "(not (=" in goal_text:
            otypes = re.findall(r"\(objectType \?\w+ (\w+)\)", goal_text)
            rtypes = re.findall(r"\(receptacleType \?\w+ (\w+)\)", goal_text)
            obj_type = otypes[0] if otypes else None
            recep_type = rtypes[0] if rtypes else None
            if obj_type:
                cond = {"objectType": obj_type, "inReceptacle": True, "count": 2}
                if recep_type:
                    cond["receptacleType"] = recep_type
                if "(isHot" in goal_text:
                    cond["isHot"] = True
                if "(isCool" in goal_text:
                    cond["isCool"] = True
                if "(isClean" in goal_text:
                    cond["isClean"] = True
                conditions.append(cond)
            return conditions
        exists_blocks = re.findall(
            r"\(and\s*((?:\s*\([^()]+\)\s*)+)\s*\)",
            goal_text
        )
        for block in exists_blocks:
            cond = {}
            otype = re.search(r"\(objectType \?\w+ (\w+)\)", block)
            rtype = re.search(r"\(receptacleType \?\w+ (\w+)\)", block)
            has_in_recep = "(inReceptacle" in block
            has_holds = "(holds" in block
            has_hot = "(isHot" in block
            has_cool = "(isCool" in block
            has_clean = "(isClean" in block
            has_toggled = "(isToggled" in block
            if otype:
                cond["objectType"] = otype.group(1)
            if rtype:
                cond["receptacleType"] = rtype.group(1)
            if has_in_recep:
                cond["inReceptacle"] = True
            if has_holds:
                cond["holds"] = True
            if has_hot:
                cond["isHot"] = True
            if has_cool:
                cond["isCool"] = True
            if has_clean:
                cond["isClean"] = True
            if has_toggled:
                cond["isToggled"] = True
            if cond:
                conditions.append(cond)
        return conditions

    def _evaluate_goal(self, conditions: list[dict]) -> bool:
        if not conditions:
            return False
        for cond in conditions:
            if not self._evaluate_single_condition(cond):
                return False
        return True

    def _evaluate_single_condition(self, cond: dict) -> bool:
        obj_type = cond.get("objectType")
        recep_type = cond.get("receptacleType")
        required_count = cond.get("count", 1)
        matching_objects = []
        if obj_type:
            for obj_id, otype in self._objects.items():
                if otype == obj_type:
                    matching_objects.append(obj_id)
        matching_receps = []
        if recep_type:
            for rec_id, rtype in self._receptacles.items():
                if rtype == recep_type:
                    matching_receps.append(rec_id)
        if cond.get("holds"):
            satisfied = 0
            for obj_id in matching_objects:
                if f"(holds agent1 {obj_id})" in self._state:
                    if self._check_obj_properties(obj_id, cond):
                        satisfied += 1
            return satisfied >= required_count
        if cond.get("inReceptacle"):
            satisfied = 0
            if matching_receps:
                for obj_id in matching_objects:
                    for rec_id in matching_receps:
                        if f"(inReceptacle {obj_id} {rec_id})" in self._state:
                            if self._check_obj_properties(obj_id, cond):
                                satisfied += 1
                            break
            else:
                for obj_id in matching_objects:
                    for pred in self._state:
                        if pred.startswith(f"(inReceptacle {obj_id} "):
                            if self._check_obj_properties(obj_id, cond):
                                satisfied += 1
                            break
            return satisfied >= required_count
        if cond.get("isToggled") and obj_type:
            for obj_id in matching_objects:
                if f"(isToggled {obj_id})" in self._state:
                    return True
            return False
        return True

    def _check_obj_properties(self, obj_id: str, cond: dict) -> bool:
        if cond.get("isHot") and f"(isHot {obj_id})" not in self._state:
            return False
        if cond.get("isCool") and f"(isCool {obj_id})" not in self._state:
            return False
        if cond.get("isClean") and f"(isClean {obj_id})" not in self._state:
            return False
        return True

    def get_available_actions(self) -> list[str]:
        loc = self._get_agent_location()
        actions = ["look", "inventory"]
        for pred in self._state:
            m = re.match(r"\(receptacleAtLocation (.+) (.+)\)", pred)
            if m:
                actions.append(f"go to {self._get_name(m.group(1))}")
        current_receps = self._get_receptacles_at_location(loc) if loc else []
        for recep_id in current_receps:
            recep_name = self._get_name(recep_id)
            if f"(openable {recep_id})" in self._state:
                if f"(opened {recep_id})" not in self._state:
                    actions.append(f"open {recep_name}")
                else:
                    actions.append(f"close {recep_name}")
            objects = self._get_objects_in_receptacle(recep_id)
            for obj_id in objects:
                if f"(pickupable {obj_id})" in self._state:
                    actions.append(f"take {self._get_name(obj_id)} from {recep_name}")
        held = self._get_held_object()
        if held:
            held_name = self._get_name(held)
            for recep_id in current_receps:
                recep_name = self._get_name(recep_id)
                actions.append(f"put {held_name} in/on {recep_name}")
                recep_type = self._receptacles.get(recep_id, "")
                if recep_type == "MicrowaveType" and f"(heatable {held})" in self._state:
                    actions.append(f"heat {held_name} with {recep_name}")
                if recep_type in ("SinkType", "SinkBasinType") and f"(cleanable {held})" in self._state:
                    actions.append(f"clean {held_name} with {recep_name}")
                if recep_type == "FridgeType" and f"(coolable {held})" in self._state:
                    actions.append(f"cool {held_name} with {recep_name}")
        for recep_id in current_receps:
            if f"(toggleable {recep_id})" in self._state:
                actions.append(f"use {self._get_name(recep_id)}")
        return actions

    def get_walkthrough(self) -> list[str]:
        if self._current_task:
            return self._current_task.walkthrough
        return []
