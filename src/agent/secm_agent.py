"""
Seam agent: Self-Evolving Causal Memory.

Core design principles:
1. Surprise-driven learning: only update memory when expectations are violated
2. Belief-strength hints: stable beliefs → strong hints, exploring → weak hints
3. Hierarchical generalization: type patterns + category abstractions
4. Sequential accumulation: memory grows across episodes
5. ReAct-style few-shot prompting for fair comparison with baselines
"""

import re
import json
from pathlib import Path
from llm.client import LLMClient
from memory.causal_graph import HierarchicalCausalGraph, BeliefState
from memory.evolution import EvolutionEngine
from memory.unit import TransitionRecord
from env.adapter import EnvironmentAdapter


# ReAct few-shot prompts.

_REACT_PROMPT_PATH = Path(__file__).parent.parent.parent / "data" / "react_prompts.json"
_REACT_PROMPTS: dict[str, str] = {}

def _load_react_prompts() -> dict[str, str]:
    global _REACT_PROMPTS
    if not _REACT_PROMPTS and _REACT_PROMPT_PATH.exists():
        with open(_REACT_PROMPT_PATH) as f:
            _REACT_PROMPTS = json.load(f)
    return _REACT_PROMPTS


MAX_THINK_STEPS = 3


class SECMAgent:
    MAX_PROMPT_CANDIDATES = 80
    MAX_PROMPT_CANDIDATE_CHARS = 5000
    MAX_HISTORY_STEPS = 6
    SCIENCEWORLD_PROMPT_CANDIDATES = 40
    SCIENCEWORLD_PROMPT_CANDIDATE_CHARS = 2500

    def __init__(self, llm: LLMClient, graph: HierarchicalCausalGraph,
                 evolution: EvolutionEngine, adapter: EnvironmentAdapter,
                 episode: int = 0, enable_llm_reflection: bool = False,
                 enable_effect_prediction: bool = True,
                 retrieval_top_k: int = 3,
                 step_retrieval_top_k: int = 3):
        self.llm = llm
        self.graph = graph
        self.evolution = evolution
        self.adapter = adapter
        self.episode = episode
        self.enable_llm_reflection = enable_llm_reflection
        self.enable_effect_prediction = enable_effect_prediction
        self.retrieval_top_k = int(retrieval_top_k)
        self.step_retrieval_top_k = int(step_retrieval_top_k)
        self._history: list[dict] = []
        self._actions_taken: list[str] = []
        self._task_type: str = ""
        self._target_obj: str = ""
        self._target_obj_type: str = ""
        self._target_recep: str = ""
        self._processing: str = ""
        self._phase: str = "search"
        self._holding: bool = False
        self._processed: bool = False
        self._visited_receps: set[str] = set()
        self._found_target: bool = False
        self._current_recep: str = ""
        self._expected_at: str = ""
        self._trajectory: list[dict] = []
        self._episode_trace: list[dict] = []
        self._start_time: float = 0
        self._last_observation: str = ""
        self._tw_recipe_ingredients: list[str] = []
        self._tw_recipe_directions: list[str] = []
        self._tw_inventory_text: str = ""
        self._tw_current_room: str = ""
        self._tw_exit_counts: dict[str, dict[str, int]] = {}

    def reset(self, task_obs: str, task_type: str = ""):
        import time as _time
        self._history = []
        self._actions_taken = []
        self._trajectory = []
        self._episode_trace = []
        self._phase = "search"
        self._holding = False
        self._processed = False
        self._visited_receps = set()
        self._found_target = False
        self._current_recep = ""
        self._expected_at = ""
        self._start_time = _time.time()
        self._last_observation = task_obs
        self._tw_recipe_ingredients = []
        self._tw_recipe_directions = []
        self._tw_inventory_text = ""
        self._tw_current_room = ""
        self._tw_exit_counts = {}

        task_info = self.adapter.parse_task(task_obs, task_type)
        self._task_type = task_info["task_type"]
        self._target_obj = task_info["target_obj"]
        self._target_obj_type = task_info["target_obj_type"]
        self._target_recep = task_info["target_recep"]
        self._processing = task_info["processing"]
        self._update_textworld_task_state(task_obs)

        few_shot = self._get_few_shot_context()
        memory_hint = self._build_initial_hint()
        react_key = self.adapter.get_few_shot_key(self._task_type)

        system_msg = self.adapter.get_system_prompt()
        method_instruction = self._method_instruction()
        if method_instruction:
            system_msg += "\n\n" + method_instruction
        if few_shot:
            system_msg += "\n" + few_shot + "\nHere is the task."
        format_rule = (
            "\n\nRules:\n"
            "- Output EXACTLY ONE line starting with \"> \" followed by your action.\n"
            "- To think first, output \"> think: <your reasoning>\" (max 3 times per turn).\n"
            f"- {self.adapter.get_action_list_text()}\n"
            "- Do NOT output anything else. Do NOT simulate environment responses."
        )
        system_msg += format_rule

        content = task_obs
        if memory_hint:
            content += memory_hint
        content += "\n> "

        self._history = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": content},
        ]

        self._episode_trace.append({
            "type": "init",
            "task_type": self._task_type,
            "target_obj": self._target_obj,
            "system_prompt_len": len(system_msg),
            "few_shot_key": react_key,
            "task_obs": task_obs,
            "memory_hint": memory_hint or None,
        })

    def act(self, observation: str | None = None, candidates: list[str] | None = None,
            reward: float = 0.0) -> str:
        hint = None
        if observation is not None and len(self._history) > 2:
            self.observe(observation, reward=reward)
            hint = self._build_step_hint(observation, candidates)
            msg = observation
            if hint:
                msg += hint
            candidate_block = self._format_candidate_block(candidates)
            if candidate_block:
                msg += candidate_block
            msg += "\n> "
            self._history.append({"role": "user", "content": msg})
        elif candidates and len(self._history) == 2:
            self._history[-1]["content"] = self._insert_candidate_block(
                self._history[-1]["content"], candidates
            )

        if candidates:
            selected = self._select_from_candidates(candidates)
            if selected:
                action, score = selected
                self._track_expectation(action)
                self._actions_taken.append(action)
                self._history.append({"role": "assistant", "content": action})
                self._episode_trace.append({
                    "type": "step",
                    "step": len(self._actions_taken),
                    "observation": observation,
                    "hint": hint,
                    "action": action,
                    "source": "memory",
                    "score": round(score, 2),
                })
                return action

        step_calls = []
        for _ in range(MAX_THINK_STEPS + 1):
            self._trim_history()
            call_idx_before = self.llm.total_calls
            act_tokens = 64 if self.llm.thinking_disabled else 1024
            response = self.llm.chat(self._history, max_tokens=act_tokens)
            if not response.strip() and not self.llm.thinking_disabled:
                response = self.llm.chat(self._history, max_tokens=act_tokens)
            action = self._parse_action(response)

            step_calls.append({
                "call_idx": call_idx_before + 1,
                "response": response.strip(),
                "parsed_action": action,
                "is_think": action.startswith("think:") or action.startswith("think "),
            })

            if action.startswith("think:") or action.startswith("think "):
                self._history.append({"role": "assistant", "content": response.strip()})
                self._history.append({"role": "user", "content": "OK."})
                continue

            if candidates:
                action = self._validate_action(action, candidates)

            self._track_expectation(action)
            self._actions_taken.append(action)
            self._history.append({"role": "assistant", "content": action})

            self._episode_trace.append({
                "type": "step",
                "step": len(self._actions_taken),
                "observation": observation,
                "hint": hint,
                "action": action,
                "source": "llm",
                "llm_calls": step_calls,
            })
            return action

        action = candidates[0] if candidates else "look"
        self._actions_taken.append(action)
        self._history.append({"role": "assistant", "content": action})
        self._episode_trace.append({
            "type": "step",
            "step": len(self._actions_taken),
            "observation": observation,
            "hint": hint,
            "action": action,
            "llm_calls": step_calls,
            "fallback": True,
        })
        return action

    def _trim_history(self):
        keep = 2 + 2 * self.MAX_HISTORY_STEPS
        if len(self._history) > keep:
            self._history = self._history[:2] + self._history[-(keep - 2):]

    def _insert_candidate_block(self, content: str, candidates: list[str]) -> str:
        candidate_block = self._format_candidate_block(candidates)
        if not candidate_block:
            return content
        marker = "\n> "
        if content.endswith(marker):
            return content[: -len(marker)] + candidate_block + marker
        return content + candidate_block + marker

    def _format_candidate_block(self, candidates: list[str] | None) -> str:
        if not candidates:
            return ""
        max_candidates, max_chars = self._candidate_prompt_limits()
        prompt_candidates = list(candidates)
        if self._candidate_text_len(prompt_candidates) > max_chars or len(prompt_candidates) > max_candidates:
            scored = [(self._score_action(action), idx, action) for idx, action in enumerate(prompt_candidates)]
            scored.sort(key=lambda item: (-item[0], item[1]))
            prompt_candidates = [action for _, _, action in scored]
        selected = []
        total_chars = 0
        for action in prompt_candidates:
            added = len(action) + 3
            if selected and total_chars + added > max_chars:
                break
            selected.append(action)
            total_chars += added
            if len(selected) >= max_candidates:
                break
        label = "Available actions for this step"
        if len(selected) < len(candidates):
            label += f" (showing {len(selected)} of {len(candidates)} most relevant)"
        actions = "\n".join(f"- {action}" for action in selected)
        if any(action.strip().lower() == "search[query]" for action in selected):
            choice_rule = (
                "For search[query], replace query with useful product keywords. "
                "Copy click actions from this list exactly."
            )
        else:
            choice_rule = "Choose exactly one action from this list. Copy the action text exactly."
        return (
            f"\n\n{label}:\n{actions}\n"
            f"{choice_rule}"
        )

    def _candidate_prompt_limits(self) -> tuple[int, int]:
        adapter_name = self.adapter.__class__.__name__.lower()
        if "scienceworld" in adapter_name or "sciworld" in adapter_name:
            return self.SCIENCEWORLD_PROMPT_CANDIDATES, self.SCIENCEWORLD_PROMPT_CANDIDATE_CHARS
        return self.MAX_PROMPT_CANDIDATES, self.MAX_PROMPT_CANDIDATE_CHARS

    @staticmethod
    def _candidate_text_len(candidates: list[str]) -> int:
        return sum(len(item) + 3 for item in candidates)

    def observe(self, observation: str, reward: float = 0.0):
        """Record the transition from the last action to this observation.

        Experiments call this for terminal observations because no subsequent
        act() call will occur after the episode ends.
        """
        if self._actions_taken:
            prev_context = self._build_context()
            transition = TransitionRecord(
                prev_obs=self._last_observation,
                action=self._actions_taken[-1],
                next_obs=observation,
                reward=reward,
                task_type=self._task_type,
                task_desc="",
                episode=self.episode,
                step=len(self._actions_taken),
                context=prev_context,
            )
            self.evolution.record_transition(transition)
        self._update_state(observation)
        if self._actions_taken:
            self._trajectory.append({
                "action": self._actions_taken[-1],
                "observation": observation,
            })
        self._last_observation = observation

    def on_episode_end(self, success: bool):
        import time as _time
        self.evolution.on_episode_end(
            self._task_type, self._actions_taken, success, self.episode
        )
        if self.enable_llm_reflection and success:
            self._reflect_on_episode(success)
        self._episode_trace.append({
            "type": "result",
            "success": success,
            "total_steps": len(self._actions_taken),
            "total_llm_calls": self.llm.total_calls,
            "duration_seconds": round(_time.time() - self._start_time, 2),
        })

    def get_episode_trace(self) -> list[dict]:
        return self._episode_trace

    # ─── Graph Training: Backward Pass ───

    def _reflect_on_episode(self, success: bool):
        traj_str = self.adapter.format_trajectory_for_reflection(self._trajectory)

        result_word = "succeeded" if success else "failed"
        prompt = [
            {"role": "system", "content": (
                "You are a learning agent. After each episode, extract what you learned "
                "as structured statements. Output ONLY lines in these exact formats:\n"
                "FOUND <object> AT <receptacle>\n"
                "NOT_FOUND <object> AT <receptacle>\n"
                "ACTION_EFFECT <action> USES <tool>\n"
                "PROCEDURE <task_type>: step1 -> step2 -> step3\n\n"
                "Use lowercase object/receptacle names and keep multiword names intact "
                "(e.g. yellow potato, red bell pepper, kitchen, fridge).\n"
                "Only output facts supported by the trajectory. For failed episodes, "
                "output locations and avoid unsupported successful procedures.\n"
                "Output NOTHING else. No explanations, no summaries, no commentary.\n\n"
                "Example input:\n"
                "I succeeded at: pick_heat_then_place_in_recep.\n"
                "- went to countertop, found egg, bread, knife\n"
                "- take egg from countertop → OK\n"
                "- heat egg with microwave → OK\n"
                "- put egg in diningtable → OK\n\n"
                "Example output:\n"
                "FOUND egg AT countertop\n"
                "FOUND bread AT countertop\n"
                "FOUND knife AT countertop\n"
                "ACTION_EFFECT heat USES microwave\n"
                "PROCEDURE pick_heat_then_place_in_recep: find object -> take object -> heat object -> put object"
            )},
            {"role": "user", "content": (
                f"I {result_word} at: {self._task_type}.\n"
                f"{traj_str}"
            )},
        ]

        try:
            reflect_tokens = 4096 if not self.llm.thinking_disabled else 512
            response = ""
            for _attempt in range(2):
                response = self.llm.chat(prompt, max_tokens=reflect_tokens)
                if response.strip():
                    break
            operations = self._apply_reflection(response, success)
            self._episode_trace.append({
                "type": "reflect",
                "success": success,
                "prompt": prompt[-1]["content"],
                "response": response,
                "operations": operations,
            })
        except Exception:
            pass

    def _apply_reflection(self, reflection: str, success: bool) -> list:
        applied = []
        for line in reflection.strip().split("\n"):
            line = line.strip()
            line = re.sub(r"^[\-*>\s]+", "", line)
            if not line:
                continue

            if re.match(r"^FOUND\s+", line, flags=re.I):
                m = re.match(r"FOUND\s+(.+?)\s+AT\s+(.+)$", line, flags=re.I)
                if m:
                    obj_type = self.adapter.obj_name_to_type(self._clean_reflection_name(m.group(1)))
                    recep_type = self.adapter.recep_name_to_type(self._clean_reflection_name(m.group(2)))
                    if obj_type and recep_type:
                        self.graph.observe(obj_type, recep_type, self.episode)
                        applied.append(("found", obj_type, recep_type))

            elif re.match(r"^NOT_FOUND\s+", line, flags=re.I):
                m = re.match(r"NOT_FOUND\s+(.+?)\s+AT\s+(.+)$", line, flags=re.I)
                if m:
                    obj_type = self.adapter.obj_name_to_type(self._clean_reflection_name(m.group(1)))
                    recep_type = self.adapter.recep_name_to_type(self._clean_reflection_name(m.group(2)))
                    if obj_type and recep_type:
                        self.graph.observe_absence(obj_type, recep_type, self.episode)
                        applied.append(("not_found", obj_type, recep_type))

            elif re.match(r"^ACTION_EFFECT\s+", line, flags=re.I):
                m = re.match(r"ACTION_EFFECT\s+(.+?)\s+USES\s+(.+)$", line, flags=re.I)
                if m:
                    action_type = self._normalize_reflection_action(m.group(1))
                    tool_name = self._clean_reflection_name(m.group(2))
                    tool_type = self.adapter.infer_appliance_type(tool_name)
                    if action_type in {"cook", "cut", "heat", "cool", "clean", "use"} and tool_type:
                        self.graph.observe_action_effect(action_type, tool_type, self.episode)
                        applied.append(("action_effect", action_type, tool_type))

            elif re.match(r"^PROCEDURE\s+", line, flags=re.I):
                m = re.match(r"PROCEDURE\s+([\w_]+):\s*(.+)", line, flags=re.I)
                if m:
                    task_type = m.group(1)
                    raw_steps = [s.strip() for s in re.split(r"\s*(?:→|->)\s*", m.group(2))]
                    steps = [self._normalize_procedure_step(s) for s in raw_steps if s]
                    steps = [s for s in steps if s]
                    if steps:
                        self.graph.record_causal_chain(
                            task_type, steps, success, self.episode
                        )
                        applied.append(("procedure", task_type, steps))

            elif line.startswith("RULE "):
                m = re.match(r"RULE\s+(.+?)\s*→\s*(.+?)(?:\s*\||$)", line)
                if m:
                    source_cat = m.group(1).strip().lower().replace(" ", "_")
                    target_cat = m.group(2).strip().lower().replace(" ", "_")
                    self.graph.add_l2_rule(
                        source_cat, target_cat, "located_at", 0.7, self.episode
                    )
                    applied.append(("rule", source_cat, target_cat))

        self.evolution._log("reflection", self.episode, {
            "task_type": self._task_type,
            "success": success,
            "operations_applied": len(applied),
            "details": applied,
        })
        return applied

    @staticmethod
    def _clean_reflection_name(text: str) -> str:
        return re.sub(r"\s+", " ", (text or "").strip().lower()).strip(" .,:;`\"'")

    @staticmethod
    def _normalize_reflection_action(action: str) -> str:
        action = SECMAgent._clean_reflection_name(action)
        return {
            "fry": "cook",
            "roast": "cook",
            "grill": "cook",
            "bake": "cook",
            "chop": "cut",
            "slice": "cut",
            "dice": "cut",
        }.get(action, action)

    def _normalize_procedure_step(self, step: str) -> str:
        s = step.strip().lower()
        if not s:
            return ""
        if any(w in s for w in ("find", "search", "look for", "locate")):
            return "find object"
        if any(w in s for w in ("take", "pick up", "grab")):
            return "take object"
        if any(w in s for w in ("put", "place")):
            return "put object"
        if "heat" in s:
            return "heat object"
        if "cool" in s:
            return "cool object"
        if "clean" in s:
            return "clean object"
        if any(w in s for w in ("cook", "fry", "roast", "grill")):
            return "cook object"
        if any(w in s for w in ("chop", "dice", "slice", "cut")):
            return "cut object"
        if "use" in s or "turn on" in s:
            return "use device"
        if "open" in s:
            return "open receptacle"
        if "go to" in s:
            m = re.match(r"go to\s+(\w+)", s)
            if m:
                return f"go to {m.group(1)}"
            return s
        return s

    # ─── Few-Shot Context ───

    def _get_few_shot_context(self, num_examples: int = 2) -> str:
        prompts = _load_react_prompts()
        if not prompts:
            return ""
        react_key = self.adapter.get_few_shot_key(self._task_type)
        secm_examples = []
        for i in range(num_examples):
            key = f"secm_{react_key}_{i}"
            if key in prompts:
                secm_examples.append(prompts[key])
        if secm_examples:
            return "\n\n".join(secm_examples)

        examples = []
        for i in range(num_examples):
            key = f"react_{react_key}_{i}"
            if key in prompts:
                examples.append(prompts[key])
        return "\n\n".join(examples)

    def _method_instruction(self) -> str:
        base = (
            "Seam may provide Memory and Hint lines learned from earlier episodes. "
            "Treat them as fallible suggestions, not as commands. First follow the "
            "current objective, current observation, inventory, and available actions. "
            "Use a memory suggestion only when it matches the current object, state, "
            "and task; ignore generic memories such as take -> holding when they do "
            "not identify the current goal object. If a memory conflicts with the "
            "current recipe, target object, or inventory state, follow the current "
            "task instead. When available actions are shown, copy one available "
            "action exactly; do not output an ideal action that is not in the list. "
            "If the desired action is absent, choose the next valid action that "
            "advances the same objective."
        )
        if self._task_type.startswith("textworld_cooking"):
            return (
                base + "\nFor TextWorld cooking, read the cookbook before collecting "
                "ingredients. The cookbook tells the recipe; it is not an ingredient. "
                "Do not take or drop the cookbook unless no examine/read action is "
                "available. Prefer the current recipe state over generic take memories: "
                "collect only recipe ingredients, perform missing recipe directions, "
                "then prepare meal and eat meal."
            )
        if self._task_type == "webshop":
            return (
                base + "\nFor WebShop, search[query] is a template: replace query "
                "with useful keywords from the instruction. Click actions must use "
                "the exact visible label from the available-action list. Do not buy "
                "until the product and selected options satisfy the instruction."
            )
        if self._task_type.startswith("textworld_treasure"):
            return (
                base + "\nFor TextWorld treasure hunt, the target object named in the "
                "objective is the only object that completes the game. Do not follow a "
                "generic take memory for a different visible object. If the target is "
                "visible, take the exact target; otherwise explore/open containers."
            )
        return base

    # ─── Hint Building (belief-strength-aware) ───

    def _build_initial_hint(self) -> str:
        hints = []
        task_hint = self._textworld_task_hint()
        if task_hint:
            hints.append(task_hint)
        if self.enable_effect_prediction:
            graph_hint = self.graph.format_hint(
                self._build_context(), top_k=self.retrieval_top_k
            )
            if graph_hint:
                hints.append(graph_hint.replace("\n[Memory]\n", "").strip())
            procedure_hint = self.graph.get_procedure_hint(self._task_type)
            if procedure_hint:
                hints.append(procedure_hint)
            if self._target_obj_type:
                loc_hint = self.graph.get_location_hint(self._target_obj_type)
                if loc_hint:
                    hints.append(loc_hint)
        if not hints:
            return ""
        return (
            "\n\n[Memory - verify before using]\n"
            "These are past-task suggestions, not commands. Use them only if they "
            "fit the current objective and observation.\n"
            + "\n".join(hints)
        )

    def _build_step_hint(self, observation: str, candidates: list[str] | None = None) -> str:
        hints = []
        task_hint = self._textworld_task_hint(candidates)
        if task_hint:
            hints.append(task_hint)
        if self.enable_effect_prediction:
            graph_hint = self.graph.format_hint(
                self._build_context(),
                candidates=candidates,
                top_k=self.step_retrieval_top_k,
            )
            if graph_hint:
                hints.append(graph_hint.replace("\n[Memory]\n", "").strip())
            procedure_hint = self.graph.get_procedure_hint(self._task_type)
            if procedure_hint:
                hints.append(procedure_hint)

        if self._phase == "search" and not self._found_target:
            if candidates and self.enable_effect_prediction:
                go_actions = [(a, self._score_action(a)) for a in candidates
                              if self.adapter.action_matches_goto(a) is not None]
                go_actions = [(a, s) for a, s in go_actions if s > 0.05]
                go_actions.sort(key=lambda x: -x[1])
                if go_actions:
                    parts = []
                    for action, score in go_actions[:3]:
                        name = self.adapter.get_recep_from_action(action)
                        if score >= 0.9:
                            parts.append(f"{name} (reliable)")
                        elif score > 0.3:
                            parts.append(f"{name} ({score:.0%})")
                        else:
                            parts.append(name)
                    hints.append(f"Try: {', '.join(parts)}")
            elif self._target_obj_type and self.enable_effect_prediction:
                locations = self.graph.query_location(self._target_obj_type, top_k=6)
                visited_bases = {r.split()[0] for r in self._visited_receps}
                if locations:
                    unvisited = [
                        (rt, conf, belief) for rt, conf, belief in locations
                        if rt.replace("Type", "").lower() not in visited_bases
                    ]
                    if unvisited:
                        parts = []
                        for rt, conf, belief in unvisited[:3]:
                            name = rt.replace("Type", "").lower()
                            if belief == BeliefState.STABLE:
                                parts.append(f"{name} (reliable)")
                            else:
                                parts.append(name)
                        hints.append(f"Try: {', '.join(parts)}")

        if self._found_target and not self._holding and self._phase == "search":
            hints.append(f"NOW: take {self._target_obj} from here")

        if self._holding and not self._processed and not self._processing and self._target_recep:
            hints.append(f"NOW: go to {self._target_recep}, then put {self._target_obj}")
        elif self._holding and self._processed and self._target_recep:
            hints.append(f"NOW: go to {self._target_recep}, then put the object there")

        if not hints:
            return ""
        return "\n[Hint - verify before using] " + " | ".join(hints)

    def _build_context(self) -> dict:
        current_recep_type = ""
        if self._current_recep:
            current_recep_type = self.adapter.recep_name_to_type(self._current_recep)
        return {
            "task_type": self._task_type,
            "target_obj": self._target_obj,
            "target_obj_type": self._target_obj_type,
            "target_recep": self._target_recep,
            "processing": self._processing,
            "phase": self._phase,
            "holding": self._holding,
            "processed": self._processed,
            "found_target": self._found_target,
            "current_recep": self._current_recep,
            "current_recep_type": current_recep_type,
            "visited_receps": sorted(self._visited_receps),
        }

    # ─── TextWorld Task-State Guidance ───

    def _update_textworld_task_state(self, observation: str):
        room = self._extract_textworld_room(observation)
        if room:
            self._tw_current_room = room
            self._current_recep = room
            self._visited_receps.add(room)
        if self._task_type.startswith("textworld_treasure") and self._textworld_target_visible(observation):
            self._found_target = True

        if not self._task_type.startswith("textworld_cooking"):
            return
        inventory = self._extract_textworld_inventory(observation)
        if inventory:
            self._tw_inventory_text = inventory

        recipe = self._extract_textworld_recipe(observation)
        if recipe:
            ingredients, directions = recipe
            if ingredients:
                self._tw_recipe_ingredients = ingredients
            if directions:
                self._tw_recipe_directions = directions

    def _textworld_task_hint(self, candidates: list[str] | None = None) -> str:
        if self._task_type.startswith("textworld_treasure"):
            if self._target_obj:
                return (
                    f"Task state: treasure target is `{self._target_obj}`. "
                    "Take only that exact target object; ignore unrelated take memories."
                )
            return "Task state: treasure target is unknown; explore/open, do not take random objects."

        if not self._task_type.startswith("textworld_cooking"):
            return ""

        if not self._tw_recipe_ingredients and not self._tw_recipe_directions:
            return (
                "Task state: recipe is unknown. Read the cookbook first; "
                "prefer `examine cookbook` over taking the cookbook or ingredients."
            )

        ingredients = ", ".join(self._tw_recipe_ingredients) or "unknown"
        directions = "; ".join(self._tw_recipe_directions) or "unknown"
        if self._is_textworld_recipe_ready():
            return (
                f"Task state: recipe ingredients [{ingredients}] and directions "
                f"[{directions}] appear satisfied in inventory. Next use `prepare meal`, "
                "then `eat meal`."
            )

        missing = self._missing_textworld_recipe_ingredients()
        if missing:
            return (
                f"Task state: recipe ingredients [{ingredients}], directions [{directions}]. "
                f"Collect missing recipe ingredient(s): {', '.join(missing)}. "
                "Do not take cookbook or unrelated food."
            )

        unfinished = self._unsatisfied_textworld_recipe_directions()
        if unfinished:
            return (
                f"Task state: recipe ingredients are collected. Next unfinished direction: "
                f"{unfinished[0]}. Do not repeat directions already visible in inventory."
            )

        return (
            f"Task state: recipe ingredients [{ingredients}], directions [{directions}]. "
            "Follow the next unfinished recipe action."
        )

    def _score_textworld_action(self, action: str) -> float | None:
        if self._task_type.startswith("textworld_treasure"):
            if self._is_take_action(action):
                if self._target_obj and self._mentions_name(action, self._target_obj):
                    return 1.0
                return -0.5 if self._target_obj else 0.0
            dest = self.adapter.action_matches_goto(action)
            if dest is not None:
                if self._is_reverse_of_last_move(dest):
                    return 0.2
                room = self._tw_current_room or "__unknown__"
                count = self._tw_exit_counts.get(room, {}).get(dest, 0)
                if count == 0:
                    return 0.85
                if count == 1:
                    return 0.35
                return 0.05
            if action.startswith("open "):
                return 0.8
            if action.startswith("examine "):
                return 0.2 if self._target_obj and self._mentions_name(action, self._target_obj) else 0.05
            return None

        if not self._task_type.startswith("textworld_cooking"):
            return None

        if action == "eat meal":
            return 1.0
        if action == "prepare meal":
            return 1.0 if self._is_textworld_recipe_ready() else 0.15
        if action.startswith("drop cookbook") or action.startswith("take cookbook"):
            return -0.5
        if action.startswith("drop "):
            return -0.2
        if action.startswith("examine cookbook"):
            return 1.0 if not self._has_textworld_recipe() else 0.15

        if not self._has_textworld_recipe():
            if self._is_take_action(action):
                return 0.0
            return None

        if self._is_take_action(action):
            if any(self._mentions_name(action, ingredient)
                   for ingredient in self._missing_textworld_recipe_ingredients()):
                return 0.95
            if "knife" in self._take_action_object(action) and self._recipe_needs_cutting():
                return 0.6
            return 0.0

        for direction in self._unsatisfied_textworld_recipe_directions():
            if self._action_satisfies_recipe_direction(action, direction):
                return 0.95

        if self._is_recipe_process_action(action):
            return -0.2
        return None

    def _blocks_textworld_memory_action(self, action: str,
                                        domain_score: float | None = None) -> bool:
        if domain_score is not None and domain_score >= 0.8:
            return False
        if self._task_type.startswith("textworld_treasure"):
            return self._is_take_action(action)
        if self._task_type.startswith("textworld_cooking"):
            if self._is_take_action(action):
                return True
            if action.startswith("drop "):
                return True
            if self._is_recipe_process_action(action):
                return True
        return False

    def _has_textworld_recipe(self) -> bool:
        return bool(self._tw_recipe_ingredients or self._tw_recipe_directions)

    def _is_textworld_recipe_ready(self) -> bool:
        if not self._has_textworld_recipe():
            return False
        if self._missing_textworld_recipe_ingredients():
            return False
        return not self._unsatisfied_textworld_recipe_directions()

    def _missing_textworld_recipe_ingredients(self) -> list[str]:
        inventory = self._tw_inventory_text.lower()
        return [
            ingredient for ingredient in self._tw_recipe_ingredients
            if ingredient and ingredient not in inventory
        ]

    def _unsatisfied_textworld_recipe_directions(self) -> list[str]:
        return [
            direction for direction in self._tw_recipe_directions
            if direction != "prepare meal" and not self._recipe_direction_satisfied(direction)
        ]

    def _recipe_direction_satisfied(self, direction: str) -> bool:
        req = self._recipe_direction_requirement(direction)
        if not req:
            return True
        _verb, ingredient, prepared_word = req
        inventory = self._tw_inventory_text.lower()
        return ingredient in inventory and prepared_word in inventory

    def _action_satisfies_recipe_direction(self, action: str, direction: str) -> bool:
        req = self._recipe_direction_requirement(direction)
        if not req:
            return False
        verb, ingredient, _prepared_word = req
        if not self._mentions_name(action, ingredient):
            return False
        if verb in {"chop", "slice", "dice"}:
            return action.startswith(verb + " ")
        if verb == "fry":
            return action.startswith(("fry ", "cook ")) and "stove" in action
        if verb in {"roast", "grill", "bake"}:
            return action.startswith((verb + " ", "cook ")) and "oven" in action
        return action.startswith(verb + " ")

    @staticmethod
    def _recipe_direction_requirement(direction: str) -> tuple[str, str, str] | None:
        clean = re.sub(r"\s+", " ", direction.lower()).strip(" .,:;")
        match = re.match(
            r"(chop|slice|dice|fry|roast|grill|bake|cook)\s+(?:the\s+)?(.+)$",
            clean,
        )
        if not match:
            return None
        verb, ingredient = match.group(1), match.group(2).strip()
        prepared = {
            "chop": "chopped",
            "slice": "sliced",
            "dice": "diced",
            "fry": "fried",
            "roast": "roasted",
            "grill": "grilled",
            "bake": "baked",
            "cook": "cooked",
        }.get(verb, verb)
        return verb, ingredient, prepared

    def _recipe_needs_cutting(self) -> bool:
        return any(
            (self._recipe_direction_requirement(direction) or ("", "", ""))[0]
            in {"chop", "slice", "dice"}
            for direction in self._tw_recipe_directions
        )

    @staticmethod
    def _extract_textworld_inventory(observation: str) -> str:
        match = re.search(r"Inventory:\s*(.*?)(?:\n\s*Score:|$)", observation or "", flags=re.I | re.S)
        if not match:
            return ""
        return re.sub(r"\s+", " ", match.group(1)).strip(" .,:;").lower()

    @staticmethod
    def _extract_textworld_recipe(observation: str) -> tuple[list[str], list[str]] | None:
        match = re.search(
            r"Ingredients:\s*(.*?)\s*Directions:\s*(.*?)(?:\n\s*>|$)",
            observation or "",
            flags=re.I | re.S,
        )
        if not match:
            return None

        def clean_lines(block: str) -> list[str]:
            lines = []
            for raw in block.splitlines():
                line = re.sub(r"\s+", " ", raw).strip(" .,:;").lower()
                if not line:
                    continue
                if line.startswith(("objective", "inventory", "score", "-=")):
                    continue
                if line.startswith(">"):
                    break
                lines.append(line)
            return lines

        ingredients = clean_lines(match.group(1))
        directions = clean_lines(match.group(2))
        return ingredients, directions

    @staticmethod
    def _extract_textworld_room(observation: str) -> str:
        matches = re.findall(r"-=\s*([^=\n]+?)\s*=-", observation or "")
        if not matches:
            return ""
        return re.sub(r"\s+", " ", matches[-1]).strip(" .,:;").lower()

    def _textworld_target_visible(self, observation: str) -> bool:
        if not self._target_obj:
            return False
        text = observation or ""
        text = re.sub(r"Objective:\s*.*?(?=\n\s*Inventory:|\n\s*Score:|$)", "", text, flags=re.I | re.S)
        text = re.sub(r"Explore the map,\s*find\s+.+?\s+finish the game\.", "", text, flags=re.I)
        return self._target_obj.lower() in text.lower()

    def _is_reverse_of_last_move(self, direction: str) -> bool:
        if not self._actions_taken:
            return False
        last = self.adapter.action_matches_goto(self._actions_taken[-1].lower().strip())
        if not last:
            return False
        return {
            "north": "south",
            "south": "north",
            "east": "west",
            "west": "east",
            "up": "down",
            "down": "up",
        }.get(last) == direction

    @staticmethod
    def _is_take_action(action: str) -> bool:
        return action.startswith(("take ", "get ", "pick up "))

    @staticmethod
    def _is_recipe_process_action(action: str) -> bool:
        return action.startswith((
            "cook ", "chop ", "slice ", "dice ", "fry ", "roast ", "grill ", "bake ",
        ))

    @staticmethod
    def _take_action_object(action: str) -> str:
        obj = re.sub(r"^(?:take|get|pick up)\s+", "", action.lower()).strip()
        obj = re.split(r"\s+from\s+", obj, maxsplit=1)[0]
        return obj.strip(" .,:;")

    @staticmethod
    def _mentions_name(text: str, name: str) -> bool:
        if not name:
            return False
        pattern = r"\b" + re.escape(name.lower()) + r"\b"
        return bool(re.search(pattern, text.lower()))

    # ─── State Tracking ───

    def _update_state(self, observation: str):
        parsed = self.adapter.parse_observation(observation)
        self._update_textworld_task_state(observation)

        if "arrived_at" in parsed:
            self._current_recep = parsed["arrived_at"]
            self._visited_receps.add(parsed["arrived_at"])
            if self._textworld_target_visible(observation):
                self._found_target = True

        if parsed.get("picked_up"):
            self._holding = True
            self._phase = "process" if self._processing else "deliver"
        elif parsed.get("put_down"):
            self._holding = False
            self._phase = "done"
        elif parsed.get("processed"):
            self._processed = True
            self._phase = "deliver"
        elif parsed.get("used_device"):
            if self._processing == "use_lamp":
                self._processed = True

    # ─── Surprise-Driven Learning ───

    def _learn_from_observation(self, observation: str):
        parsed = self.adapter.parse_observation(observation)

        if "arrived_at" not in parsed:
            return

        recep_name = parsed["arrived_at"]
        recep_type = self.adapter.recep_name_to_type(recep_name)
        if not recep_type:
            return

        found_obj_types = set()
        for obj_name in parsed.get("objects_here", []):
            obj_type = self.adapter.obj_name_to_type(obj_name)
            if obj_type:
                found_obj_types.add(obj_type)
                self.graph.observe(obj_type, recep_type, self.episode)

        if (self._phase == "search"
                and self._target_obj_type
                and self._expected_at
                and recep_type == self._expected_at
                and self._target_obj_type not in found_obj_types):
            self.graph.observe_absence(self._target_obj_type, recep_type, self.episode)
        self._expected_at = ""

    def _track_expectation(self, action: str):
        dest = self.adapter.action_matches_goto(action)
        if dest and self._task_type.startswith("textworld_treasure"):
            room = self._tw_current_room or "__unknown__"
            counts = self._tw_exit_counts.setdefault(room, {})
            counts[dest] = counts.get(dest, 0) + 1
        if dest and self._phase == "search":
            if not self.enable_effect_prediction:
                return
            recep_type = self.adapter.recep_name_to_type(dest)
            if recep_type and self._target_obj_type:
                locations = self.graph.query_location(self._target_obj_type, top_k=3)
                for rt, conf, belief in locations:
                    if rt == recep_type and belief == BeliefState.STABLE:
                        self._expected_at = recep_type
                        break

    # ─── Candidate Action Selection ───

    def _score_action(self, action: str) -> float:
        a = action.lower().strip()
        domain_score = self._score_textworld_action(a)
        if domain_score is not None and (domain_score >= 0.8 or domain_score < 0):
            return domain_score

        dest = self.adapter.action_matches_goto(a)
        if dest is not None and self._current_recep and dest == self._current_recep:
            return -1.0

        task_state = {
            "goal": self._task_type,
            "holding": self._holding,
            "processed": self._processed,
            "found_target": self._found_target,
            "processing": self._processing,
            "target_obj_type": self._target_obj_type,
        }
        if self.enable_effect_prediction:
            memory_score = self.graph.score_action(a, self._build_context())
            if memory_score >= 0.8 and not self._blocks_textworld_memory_action(a, domain_score):
                return memory_score

        suggestion = self.graph.query_next_action(task_state) if self.enable_effect_prediction else None

        if suggestion and not self._blocks_textworld_memory_action(a, domain_score):
            return self._score_by_suggestion(a, suggestion)

        fallback = self._score_fallback(a)
        if domain_score is not None:
            return max(domain_score, fallback)
        return fallback

    def _score_by_suggestion(self, action: str, suggestion: tuple[str, str, float]) -> float:
        action_type, target_type, conf = suggestion

        if action_type == "goto":
            dest = self.adapter.action_matches_goto(action)
            if dest is not None:
                if dest in self._visited_receps:
                    return -1.0
                recep_type = self.adapter.recep_name_to_type(dest)
                if recep_type == target_type:
                    return conf
                return 0.1
            if self.adapter.action_matches_take(action, self._target_obj):
                return 1.0
            if action.startswith("open "):
                return 0.15
            return 0.0

        elif action_type == "take":
            if self.adapter.action_matches_take(action, self._target_obj):
                return 1.0
            return 0.0

        elif action_type == "goto_appliance":
            dest = self.adapter.action_matches_goto(action)
            if dest is not None:
                recep_type = self.adapter.recep_name_to_type(dest)
                if recep_type == target_type:
                    return conf
                return 0.05
            if self.adapter.action_matches_process(action, self._processing):
                return 1.0
            return 0.0

        elif action_type == "deliver":
            at_target = (self._target_recep and self._current_recep
                         and self._target_recep in self._current_recep)
            if self.adapter.action_matches_put(action):
                return 1.0 if at_target else 0.5
            dest = self.adapter.action_matches_goto(action)
            if dest is not None and self._target_recep and self._target_recep in action:
                return 0.95
            if dest is not None:
                return 0.05
            return 0.0

        return 0.05

    def _score_fallback(self, action: str) -> float:
        if not self._found_target and not self._holding:
            dest = self.adapter.action_matches_goto(action)
            if dest is not None:
                if dest in self._visited_receps:
                    return -1.0
                return 0.1
            if self.adapter.action_matches_take(action, self._target_obj):
                return 1.0
            if action.startswith("open "):
                return 0.15
            return 0.0

        if self._found_target and not self._holding:
            if self.adapter.action_matches_take(action, self._target_obj):
                return 1.0
            return 0.0

        if self._holding and not self._processed and self._processing:
            appliance = self.adapter.get_cold_start_appliance(self._processing)
            dest = self.adapter.action_matches_goto(action)
            if dest is not None and appliance and appliance in action:
                return 0.95
            if self.adapter.action_matches_process(action, self._processing):
                return 1.0
            if dest is not None:
                return 0.05
            return 0.0

        if self._holding and (self._processed or not self._processing):
            at_target = (self._target_recep and self._current_recep
                         and self._target_recep in self._current_recep)
            if self.adapter.action_matches_put(action):
                return 1.0 if at_target else 0.5
            dest = self.adapter.action_matches_goto(action)
            if dest is not None and self._target_recep and self._target_recep in action:
                return 0.95
            if dest is not None:
                return 0.05
            return 0.0

        return 0.05

    def _select_from_candidates(self, candidates: list[str]) -> tuple[str, float] | None:
        is_template = getattr(self.adapter, "is_action_template", None)
        selectable = [a for a in candidates if not (is_template and is_template(a))]
        scored = [(a, self._score_action(a)) for a in selectable]
        scored.sort(key=lambda x: -x[1])
        if scored and scored[0][1] >= 0.8:
            best_action = scored[0][0]
            if self._actions_taken and best_action == self._actions_taken[-1]:
                return None
            if len(self._actions_taken) >= 2 and best_action == self._actions_taken[-2]:
                return None
            return (best_action, scored[0][1])
        return None

    def _validate_action(self, action: str, candidates: list[str]) -> str:
        action = self.adapter.normalize_action(action, candidates)
        action_lower = action.lower().strip()
        domain_score = self._score_textworld_action(action_lower)
        if domain_score is not None and domain_score < 0:
            return self._best_allowed_textworld_candidate(candidates)
        candidates_lower = {c.lower(): c for c in candidates}
        if action_lower in candidates_lower:
            return candidates_lower[action_lower]
        if self.adapter.allow_unlisted_action(action, candidates):
            return action
        for cl, c in candidates_lower.items():
            if cl.startswith(action_lower) or action_lower.startswith(cl):
                return c
        action_words = set(action_lower.split())
        action_verb = self._action_verb(action_lower)
        best_match, best_overlap = None, 0
        for cl, c in candidates_lower.items():
            if not self._compatible_action_verbs(action_verb, self._action_verb(cl), action_lower, cl):
                continue
            overlap = len(action_words & set(cl.split()))
            if overlap > best_overlap:
                best_overlap = overlap
                best_match = c
        if best_match and best_overlap >= 2:
            return best_match
        scored = sorted(candidates, key=lambda c: -self._score_action(c))
        return scored[0]

    def _best_allowed_textworld_candidate(self, candidates: list[str]) -> str:
        scored = []
        for idx, candidate in enumerate(candidates):
            low = candidate.lower().strip()
            domain_score = self._score_textworld_action(low)
            if domain_score is not None and domain_score < 0:
                continue
            scored.append((self._score_action(candidate), -idx, candidate))
        if not scored:
            return candidates[0] if candidates else "look"
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return scored[0][2]

    @staticmethod
    def _action_verb(action: str) -> str:
        parts = action.split()
        return parts[0] if parts else ""

    @staticmethod
    def _compatible_action_verbs(action_verb: str, candidate_verb: str,
                                 action: str, candidate: str) -> bool:
        if action_verb == candidate_verb:
            return True
        directions = {"north", "south", "east", "west", "up", "down"}
        if action_verb == "go" and candidate_verb in directions:
            return candidate_verb in action.split()
        if candidate_verb == "go" and action_verb in directions:
            return action_verb in candidate.split()
        aliases = (
            {"take", "get", "pick"},
            {"examine", "look", "read", "inspect", "check"},
            {"put", "place", "insert"},
        )
        return any(action_verb in group and candidate_verb in group for group in aliases)

    # ─── Action Parsing ───

    def _parse_action(self, response: str) -> str:
        response = response.strip()
        if not response:
            return "look"
        valid_cmds = self.adapter.get_valid_commands()
        first_think = ""
        valid_actions: list[str] = []
        lines = response.split("\n")
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                line = line[1:].strip()
            line = re.sub(r"^(action:\s*)", "", line, flags=re.I)
            line = self.adapter.normalize_action(line)
            low = line.lower()
            if low.startswith("think:") or low.startswith("think "):
                if not first_think:
                    first_think = line
                continue
            if any(low.startswith(cmd) for cmd in valid_cmds):
                valid_actions.append(low)
        if valid_actions:
            return valid_actions[-1]
        if first_think:
            return first_think
        return "look"


SEAMAgent = SECMAgent
