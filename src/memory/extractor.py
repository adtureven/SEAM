"""
Transition-level extraction for condition-action-effect memories.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from memory.unit import ActionSchema, CausalMemoryUnit, Condition, Effect, TransitionRecord

if TYPE_CHECKING:
    from env.adapter import EnvironmentAdapter


class TransitionExtractor:
    """Extract hypothesis memories from (o_t, a_t, o_{t+1}, r_t, task)."""

    def __init__(self, adapter: "EnvironmentAdapter"):
        self.adapter = adapter

    def extract(self, transition: TransitionRecord) -> list[CausalMemoryUnit]:
        action = self.parse_action_schema(transition.action)
        if not action.action_type:
            return []

        condition = self._condition_from_transition(transition, action)
        effects = self._effects_from_transition(transition, action)
        memories = []
        for effect in effects:
            memories.append(CausalMemoryUnit(
                condition=condition,
                action=action,
                effect=effect,
                source="transition",
            ))
        return memories

    def parse_action_schema(self, action: str) -> ActionSchema:
        a = action.lower().strip()
        if not a:
            return ActionSchema("")

        # WebShop actions encode their arguments inside brackets. Search terms
        # are intentionally omitted from the schema so the memory can generalize
        # across different queries; clicks are grouped by control role.
        search = re.fullmatch(r"search\[(.*)\]", a)
        if search:
            return ActionSchema("search", raw=a)
        click = re.fullmatch(r"click\[(.*)\]", a)
        if click:
            label = click.group(1).strip()
            click_type = getattr(self.adapter, "click_type", None)
            role = click_type(label) if click_type else "clickable"
            return ActionSchema("click", object_type=role, raw=a)

        dest = self.adapter.action_matches_goto(a)
        if dest is not None:
            return ActionSchema(
                action_type="goto",
                receptacle_type=self.adapter.recep_name_to_type(dest),
                raw=a,
            )

        if a.startswith("open "):
            recep = a[len("open "):].strip()
            return ActionSchema("open", receptacle_type=self.adapter.recep_name_to_type(recep), raw=a)
        if a.startswith("close "):
            recep = a[len("close "):].strip()
            return ActionSchema("close", receptacle_type=self.adapter.recep_name_to_type(recep), raw=a)
        if a.startswith("use "):
            obj = a[len("use "):].strip()
            return ActionSchema(
                "use",
                object_type=self.adapter.obj_name_to_type(obj) or self.adapter.recep_name_to_type(obj),
                raw=a,
            )
        if a in ("look", "inventory", "prepare meal", "eat meal"):
            return ActionSchema(a.replace(" ", "_"), raw=a)

        take = re.match(r"take\s+(.+?)\s+from\s+(.+)", a)
        if take:
            obj, recep = take.group(1).strip(), take.group(2).strip()
            return ActionSchema(
                "take",
                object_type=self.adapter.obj_name_to_type(obj),
                receptacle_type=self.adapter.recep_name_to_type(recep),
                raw=a,
            )

        put = re.match(r"(?:put|move)\s+(.+?)\s+(?:to|in/on|in|on)\s+(.+)", a)
        if put:
            obj, recep = put.group(1).strip(), put.group(2).strip()
            return ActionSchema(
                "put",
                object_type=self.adapter.obj_name_to_type(obj),
                receptacle_type=self.adapter.recep_name_to_type(recep),
                raw=a,
            )

        process = re.match(r"(heat|clean|cool|cook|chop|dice|slice|fry|roast|grill)\s+(.+?)(?:\s+with\s+(.+))?$", a)
        if process:
            verb, obj, tool = process.group(1), process.group(2).strip(), (process.group(3) or "").strip()
            action_type = {
                "fry": "cook",
                "roast": "cook",
                "grill": "cook",
                "chop": "cut",
                "dice": "cut",
                "slice": "cut",
            }.get(verb, verb)
            tool_type = self.adapter.infer_appliance_type(tool) or self.adapter.recep_name_to_type(tool)
            return ActionSchema(
                action_type,
                object_type=self.adapter.obj_name_to_type(obj),
                tool_type=tool_type,
                raw=a,
            )

        if a.startswith("drop "):
            obj = a[len("drop "):].strip()
            return ActionSchema("drop", object_type=self.adapter.obj_name_to_type(obj), raw=a)

        if a.startswith("examine "):
            obj = a[len("examine "):].strip()
            return ActionSchema(
                "examine",
                object_type=self.adapter.obj_name_to_type(obj) or self.adapter.recep_name_to_type(obj),
                raw=a,
            )

        return ActionSchema(a.split()[0], raw=a)

    def _condition_from_transition(self, transition: TransitionRecord,
                                   action: ActionSchema) -> Condition:
        ctx = transition.context
        current_recep_type = ctx.get("current_recep_type", "")
        if not current_recep_type:
            parsed_prev = self.adapter.parse_observation(transition.prev_obs)
            if "arrived_at" in parsed_prev:
                current_recep_type = self.adapter.recep_name_to_type(parsed_prev["arrived_at"])

        predicates = []
        if "closed" in transition.prev_obs.lower():
            predicates.append("closed_receptacle_visible")
        if action.action_type in ("heat", "clean", "cool", "cook", "cut"):
            predicates.append(f"requires_tool:{action.action_type}")
        if action.action_type == "take":
            predicates.append("object_visible")
        if action.action_type == "put":
            predicates.append("holding_object")

        return Condition(
            task_type=transition.task_type or ctx.get("task_type", ""),
            target_obj_type=ctx.get("target_obj_type", ""),
            current_recep_type=current_recep_type,
            holding=ctx.get("holding"),
            processed=ctx.get("processed"),
            processing=ctx.get("processing", ""),
            predicates=tuple(sorted(set(predicates))),
        )

    def _effects_from_transition(self, transition: TransitionRecord,
                                 action: ActionSchema) -> list[Effect]:
        parsed_next = self.adapter.parse_observation(transition.next_obs)
        effects: list[Effect] = []
        low_next = transition.next_obs.lower()

        if action.action_type == "search":
            if parsed_next.get("page") == "search_results":
                effects.append(Effect(
                    "page_transition", value="search_results",
                    text="search results page is displayed",
                ))
            elif parsed_next.get("error"):
                effects.append(Effect("search_failed", text="search returned no results"))
            return effects

        if action.action_type == "click":
            if parsed_next.get("task_complete"):
                effects.append(Effect(
                    "task_complete", value="webshop_success",
                    text="the selected product satisfies the task",
                ))
            elif action.object_type == "buy":
                # An unsuccessful purchase should not become a reusable
                # prediction that buying any product completes the task.
                return effects
            elif parsed_next.get("page"):
                effects.append(Effect(
                    "page_transition", value=parsed_next["page"],
                    text=f"page changes to {parsed_next['page']}",
                ))
            elif parsed_next.get("error"):
                effects.append(Effect("click_failed", text="click did not reach a page"))
            return effects

        if action.action_type == "goto" and "arrived_at" in parsed_next:
            target_obj_type = transition.context.get("target_obj_type", "")
            visible_types = {
                self.adapter.obj_name_to_type(obj)
                for obj in parsed_next.get("objects_here", [])
            }
            visible_types = {v for v in visible_types if v}
            for obj_type in visible_types:
                effects.append(Effect(
                    "observe",
                    object_type=obj_type,
                    receptacle_type=action.receptacle_type,
                    text=f"{obj_type} becomes visible",
                ))
            if target_obj_type and target_obj_type in visible_types:
                effects.append(Effect(
                    "observe_target",
                    object_type=target_obj_type,
                    receptacle_type=action.receptacle_type,
                    text=f"target {target_obj_type} becomes visible",
                ))

        if action.action_type == "open" and ("you open" in low_next or " is open" in low_next):
            effects.append(Effect(
                "opened",
                receptacle_type=action.receptacle_type,
                text=f"{action.receptacle_type} becomes open",
            ))

        if action.action_type == "take" and parsed_next.get("picked_up"):
            effects.append(Effect(
                "holding",
                object_type=action.object_type,
                receptacle_type=action.receptacle_type,
                text=f"agent holds {action.object_type}",
            ))

        if action.action_type in ("put", "drop") and parsed_next.get("put_down"):
            effects.append(Effect(
                "placed",
                object_type=action.object_type,
                receptacle_type=action.receptacle_type,
                text=f"{action.object_type} is placed at {action.receptacle_type}",
            ))

        proc_type = parsed_next.get("processed")
        if proc_type:
            effects.append(Effect(
                "processed",
                object_type=action.object_type,
                receptacle_type=action.tool_type or action.receptacle_type,
                value=proc_type,
                text=f"{action.object_type} becomes {proc_type}",
            ))

        if action.action_type == "use" and parsed_next.get("used_device"):
            effects.append(Effect(
                "toggled",
                object_type=action.object_type,
                value="on",
                text=f"{action.object_type} is toggled",
            ))

        if "you have completed the task" in low_next:
            effects.append(Effect(
                "task_complete",
                object_type=transition.context.get("target_obj_type", ""),
                value=transition.task_type,
                text="task is completed",
            ))

        return effects
