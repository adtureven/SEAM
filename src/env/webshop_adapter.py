"""SEAM adapter for WebShop's search/click action space."""

from __future__ import annotations

import re

from env.adapter import EnvironmentAdapter


class WebShopAdapter(EnvironmentAdapter):
    PAGE_TYPES = ("home", "search_results", "item_page", "item_sub_page", "done")

    def parse_observation(self, obs: str) -> dict:
        text = str(obs or "")
        low = text.lower()
        result: dict = {}
        page = re.search(r"webshop page:\s*([a-z_]+)", low)
        if page:
            result["page"] = page.group(1)
            result["arrived_at"] = page.group(1)
        reward = re.search(r"reward:\s*(-?\d+(?:\.\d+)?)", low)
        if reward:
            result["reward"] = float(reward.group(1))
        done = re.search(r"done:\s*(true|false)", low)
        if done:
            result["done"] = done.group(1) == "true"
        result["task_complete"] = bool(
            result.get("done") and result.get("reward", 0.0) >= 0.99
        )
        if "no results" in low or "no products" in low:
            result["error"] = "No products found for this search."
        return result

    def obj_name_to_type(self, name: str) -> str:
        return self._canonical_type(name)

    def recep_name_to_type(self, name: str) -> str:
        value = (name or "").strip().lower().replace(" ", "_")
        return f"WebPage:{value}" if value in self.PAGE_TYPES else ""

    def parse_task(self, obs: str, task_type: str = "") -> dict:
        match = re.search(r"Instruction:\s*(.+)", str(obs), flags=re.I)
        desc = match.group(1).strip() if match else ""
        return {
            "task_type": task_type or "webshop",
            "target_obj": desc,
            # Product instructions are open-ended descriptions, not a fixed
            # object ontology. Keep the task key broad so learned transitions
            # can transfer between instructions.
            "target_obj_type": "",
            "target_recep": "",
            "processing": "",
        }

    def action_matches_goto(self, action: str) -> str | None:
        return None

    def action_matches_take(self, action: str, target_obj: str) -> bool:
        return False

    def action_matches_process(self, action: str, process_type: str) -> bool:
        return False

    def action_matches_put(self, action: str) -> bool:
        return False

    def action_matches_use_device(self, action: str) -> bool:
        return (action or "").strip().lower().startswith("click[")

    @staticmethod
    def click_type(label: str) -> str:
        low = (label or "").strip().lower()
        if low in {"buy now", "buy now!", "purchase"}:
            return "buy"
        if re.fullmatch(r"[a-z0-9]{8,20}", low):
            return "product"
        if low in {"next", "next page", "next >", "prev", "previous", "back to search"}:
            return "navigation"
        if low in {"description", "features", "reviews"}:
            return "product_info"
        return "option"

    def normalize_action(self, action: str, candidates: list[str] | None = None) -> str:
        normalized = re.sub(r"\s+", " ", (action or "").strip())
        normalized = re.sub(r"^(?:action|final action)\s*:\s*", "", normalized, flags=re.I)
        normalized = normalized.strip().strip("`\"'")
        if candidates:
            low = normalized.lower()
            for candidate in candidates:
                if candidate.lower() == low:
                    return candidate
        return normalized

    def allow_unlisted_action(self, action: str, candidates: list[str] | None = None) -> bool:
        low = (action or "").strip().lower()
        return bool(re.fullmatch(r"search\[[^\]]+\]", low) and low != "search[query]")

    @staticmethod
    def is_action_template(action: str) -> bool:
        low = (action or "").strip().lower()
        return low == "search[query]" or low in {"click[buy now]", "click[buy now!]"}

    def extract_action_effect(self, action: str) -> tuple[str, str] | None:
        schema = self._parse_action(action)
        if schema is None:
            return None
        return schema

    def infer_appliance_type(self, appliance_name: str) -> str:
        return self._canonical_type(appliance_name)

    def get_object_category(self, obj_type: str) -> str | None:
        return "webshop_action" if obj_type else None

    def get_receptacle_category(self, recep_type: str) -> str | None:
        return "webpage" if recep_type.startswith("WebPage:") else None

    def get_cold_start_appliance(self, process_type: str) -> str:
        return ""

    def get_system_prompt(self) -> str:
        return (
            "You are an agent completing product-finding tasks on WebShop. Read the "
            "instruction, search with useful keywords, inspect products and options, "
            "and click the available controls. Finish by clicking Buy Now only when "
            "the selected product and options satisfy the instruction."
        )

    def get_few_shot_key(self, task_type: str) -> str:
        return "webshop"

    def get_valid_commands(self) -> tuple[str, ...]:
        return ("search[", "click[")

    def get_action_list_text(self) -> str:
        return (
            "Actions use exactly one of these forms: search[keywords] or "
            "click[visible clickable text]. Choose click text from the available "
            "actions listed for the current page."
        )

    def format_trajectory_for_reflection(self, trajectory: list[dict]) -> str:
        lines = []
        for step in trajectory:
            action = step.get("action", "")
            observation = re.sub(r"\s+", " ", step.get("observation", "")).strip()
            lines.append(f"- {action} -> {observation[:400]}")
        return "\n".join(lines)

    def get_recep_from_action(self, action: str) -> str:
        return ""

    @staticmethod
    def _parse_action(action: str) -> tuple[str, str] | None:
        match = re.fullmatch(r"\s*(search|click)\[(.*)\]\s*", action or "", flags=re.I)
        if not match:
            return None
        kind, value = match.group(1).lower(), match.group(2).strip()
        if kind == "search":
            return "search", value
        label = value.lower().strip()
        if label in {"buy now", "buy now!", "purchase"}:
            label = "buy_now"
        elif label in {"next", "next page", "next >"}:
            label = "next_page"
        elif label in {"prev", "previous", "back to search"}:
            label = "back"
        elif re.fullmatch(r"[a-z0-9]{8,20}", label):
            label = "product"
        else:
            label = WebShopAdapter._canonical_type(label)
        return "click", label

    @staticmethod
    def _canonical_type(value: str) -> str:
        words = re.findall(r"[a-z0-9]+", (value or "").lower())
        return " ".join(words[:6])
