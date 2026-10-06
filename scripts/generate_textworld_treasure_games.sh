#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-data/textworld_treasure/games}"
HOME_DIR="${TEXTWORLD_HOME:-data/textworld/.home}"
GAMES_PER_LEVEL="${TEXTWORLD_GAMES_PER_LEVEL:-100}"
FORCE="${FORCE:-0}"

mkdir -p "$OUTPUT_DIR" "$HOME_DIR"

MANIFEST="$OUTPUT_DIR/manifest.tsv"
printf 'difficulty\tindex\tseed\tlevel\tfile\n' > "$MANIFEST"

generate_level() {
  local difficulty="$1"
  local level="$2"
  local seed_base="$3"
  local level_dir="$OUTPUT_DIR/$difficulty"
  mkdir -p "$level_dir"

  for idx in $(seq 0 $((GAMES_PER_LEVEL - 1))); do
    local seed=$((seed_base + idx))
    local file="$level_dir/textworld_treasure_${difficulty}_$(printf '%04d' "$idx").z8"
    if [[ "$FORCE" == "1" || ! -s "$file" ]]; then
      HOME="$HOME_DIR" tw-make tw-treasure_hunter \
        --level "$level" \
        --seed "$seed" \
        --output "$file" \
        --silent \
        --force
    fi
    printf '%s\t%s\t%s\t%s\t%s\n' "$difficulty" "$idx" "$seed" "$level" "$file" >> "$MANIFEST"
  done
}

generate_level easy 5 700000
generate_level medium 15 800000
generate_level hard 25 900000

echo "TextWorld Treasure Hunter games ready under $OUTPUT_DIR"
echo "Manifest: $MANIFEST"
