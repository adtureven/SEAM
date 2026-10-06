#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-data/textworld_cooking/games}"
HOME_DIR="${TEXTWORLD_HOME:-data/textworld/.home}"
GAMES_PER_LEVEL="${TEXTWORLD_GAMES_PER_LEVEL:-100}"
FORCE="${FORCE:-0}"

mkdir -p "$OUTPUT_DIR" "$HOME_DIR"

MANIFEST="$OUTPUT_DIR/manifest.tsv"
printf 'difficulty\tindex\tseed\trecipe_seed\trecipe\ttake\tgo\tflags\tfile\n' > "$MANIFEST"

generate_level() {
  local difficulty="$1"
  local recipe="$2"
  local take="$3"
  local go="$4"
  local seed_base="$5"
  shift 5
  local flags=("$@")
  local level_dir="$OUTPUT_DIR/$difficulty"
  mkdir -p "$level_dir"

  for idx in $(seq 0 $((GAMES_PER_LEVEL - 1))); do
    local seed=$((seed_base + idx))
    local file="$level_dir/textworld_cooking_${difficulty}_$(printf '%04d' "$idx").z8"
    if [[ "$FORCE" == "1" || ! -s "$file" ]]; then
      HOME="$HOME_DIR" tw-make tw-cooking \
        --recipe "$recipe" \
        --take "$take" \
        --go "$go" \
        --seed "$seed" \
        --recipe-seed "$seed" \
        "${flags[@]}" \
        --output "$file" \
        --silent \
        --force
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$difficulty" "$idx" "$seed" "$seed" "$recipe" "$take" "$go" "${flags[*]}" "$file" >> "$MANIFEST"
  done
}

generate_level easy 1 1 1 400000
generate_level medium 2 2 6 500000 --open --cook
generate_level hard 3 3 9 600000 --open --cook --cut --drop

echo "TextWorld Cooking games ready under $OUTPUT_DIR"
echo "Manifest: $MANIFEST"
