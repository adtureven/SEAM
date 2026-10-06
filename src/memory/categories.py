"""
Category system for Seam.

Categories enable L2 abstraction: when multiple types in the same category
share a pattern, we generalize to category-level rules.

Category data is provided by EnvironmentAdapter — no environment-specific
data is hardcoded here.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from env.adapter import EnvironmentAdapter


def get_object_category(obj_type: str, adapter: "EnvironmentAdapter | None" = None) -> str | None:
    if adapter:
        return adapter.get_object_category(obj_type)
    return None


def get_receptacle_category(recep_type: str, adapter: "EnvironmentAdapter | None" = None) -> str | None:
    if adapter:
        return adapter.get_receptacle_category(recep_type)
    return None
