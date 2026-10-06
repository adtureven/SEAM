"""Dataset preparation helper for the Seam code appendix.

The script avoids downloading large benchmark files automatically. It validates
local dependencies and prints exact commands/locations so the appendix remains
portable and free of machine-specific paths.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def check_alfworld() -> int:
    expected = ROOT / "data" / "alfworld" / "json_2.1.1" / "valid_unseen"
    print("ALFWorld")
    print(f"Expected data directory: {expected.relative_to(ROOT)}")
    print("Install and download ALFWorld following the official project instructions.")
    print("Then copy or symlink the json_2.1.1 directory to data/alfworld/json_2.1.1.")
    print(f"Current status: {'found' if expected.exists() else 'missing'}")
    return 0 if expected.exists() else 1


def check_textworld(kind: str) -> int:
    module_ok = has_module("textworld")
    script = (
        "scripts/generate_textworld_cooking_games.sh"
        if kind == "cooking"
        else "scripts/generate_textworld_treasure_games.sh"
    )
    print(f"TextWorld {kind}")
    print(f"textworld package: {'found' if module_ok else 'missing'}")
    print(f"Generation command: bash {script}")
    print("The command generates 100 games for each of easy, medium, and hard.")
    return 0 if module_ok else 1


def check_scienceworld() -> int:
    module_ok = has_module("scienceworld")
    indices = ROOT / "data" / "scienceworld_270_indices.json"
    print("ScienceWorld")
    print(f"scienceworld package: {'found' if module_ok else 'missing'}")
    print(f"270-instance index file: {indices.relative_to(ROOT)}")
    print("Run with --env scienceworld and max_steps=100.")
    return 0 if module_ok and indices.exists() else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=["alfworld", "textworld-cooking", "textworld-treasure", "scienceworld", "all"],
        default="all",
    )
    args = parser.parse_args()

    statuses = []
    if args.dataset in ("alfworld", "all"):
        statuses.append(check_alfworld())
    if args.dataset in ("textworld-cooking", "all"):
        statuses.append(check_textworld("cooking"))
    if args.dataset in ("textworld-treasure", "all"):
        statuses.append(check_textworld("treasure"))
    if args.dataset in ("scienceworld", "all"):
        statuses.append(check_scienceworld())
    return 0 if all(status == 0 for status in statuses) else 1


if __name__ == "__main__":
    raise SystemExit(main())
