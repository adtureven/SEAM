# Seam Code Appendix

This directory contains a cleaned, self-contained code appendix for **Seam:
Self-Evolving Causal Memory**. It includes the core algorithm, baseline
implementations, evaluation drivers, dataset preparation scripts, prompts, and
paper hyperparameters. It intentionally excludes result files, model weights,
raw benchmark data, API keys, and machine-specific paths.

## Layout

```text
configs/                  Sanitized local/API config templates
experiments/              Sequential evaluation and ablation entry points
scripts/                  Dataset preparation/generation helpers
src/agent/                Seam agent and baseline agents
src/env/                  ALFWorld, TextWorld, and ScienceWorld wrappers
src/memory/               Causal memory units, graph, verification, evolution
src/llm/                  OpenAI-compatible local/API client
```

The implementation class is still named `SECMAgent` in one source file for
backward compatibility, and is exported as `SEAMAgent`. All public experiment
commands and output method names use `seam`.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-optional.txt
```

For local inference, start an OpenAI-compatible vLLM server, for example:

```bash
vllm serve <MODEL_DIR> --served-model-name qwen3-4b --port 8000
```

Then edit `configs/seam_aaai2026.yaml` only if your local endpoint or model
name differs. For external APIs, use `configs/api_template.yaml` and export the
key through an environment variable; do not write keys into config files.

## Dataset Preparation

See `docs/DATASETS.md` for details. Minimal commands:

```bash
python scripts/prepare_datasets.py --dataset alfworld
bash scripts/generate_textworld_cooking_games.sh
bash scripts/generate_textworld_treasure_games.sh
python scripts/prepare_datasets.py --dataset scienceworld
```

## Main Evaluation

ALFWorld Seam:

```bash
python experiments/run_sequential.py 134 seam 50 configs/seam_aaai2026.yaml --env alfworld
```

ScienceWorld Seam on the 270-instance subset:

```bash
python experiments/run_sequential.py 270 seam 100 configs/seam_aaai2026.yaml --env scienceworld
```

WebShop (text environment): clone and set up the upstream Princeton WebShop
repository separately, then set `webshop.repo_path` in your config to that
checkout. The upstream setup downloads its product and goal data and builds the
search index. For example:

```bash
git clone https://github.com/princeton-nlp/WebShop.git ../WebShop
# Follow the upstream setup instructions in ../WebShop.
# Set webshop.repo_path to ../WebShop in configs/default.yaml.
python experiments/run_sequential.py 500 seam 15 configs/default.yaml --env webshop
```

WebShop defaults to all products (`num_products: null`), the official 500-task
test split (`split: test`), and a 15-step episode limit. The official split uses
goal indices 0–499 after the upstream fixed shuffle (seed 233), following
https://github.com/princeton-nlp/WebShop/blob/master/baseline_models/env.py.
Keep `limit_goals: -1` and `human_goals: false` to preserve this split.
The full upstream product data and search index must be installed.
`webshop.num_episodes: 500` is used when the episode count is omitted or set to 0.

TextWorld Cooking curriculum:

```bash
python experiments/run_sequential.py 300 seam 50 configs/seam_aaai2026.yaml --env textworld_cooking_curriculum
```

TextWorld Treasure Hunter curriculum:

```bash
python experiments/run_sequential.py 300 seam 50 configs/seam_aaai2026.yaml --env textworld_treasure_curriculum
```

Baselines use the same entry point. Replace `seam` with `react`, `reflexion`,
`amem`, `mem0`, `expel`, `awm`, `synapse`, `swiftsage`, `rag`, `autoguide`, or
`rap`. Use `baselines` to run all registered baselines sequentially.

## Ablation Evaluation

Run all five ALFWorld ablations:

```bash
python experiments/run_ablation.py all 134 50 configs/seam_aaai2026.yaml
```

Run one ablation:

```bash
python experiments/run_ablation.py no_effect_prediction 134 50 configs/seam_aaai2026.yaml
```

The ablated modules are self-evolution, evidence-based verification, causal
abstraction, memory revision, and effect prediction.

## Outputs

Runs create a new directory under `outputs/` with:

```text
summary.json              Aggregate success, score, steps, token/call metrics
episodes/                 Per-episode traces
memory/                   Final memory graph and evolution logs
llm_calls.jsonl           Redacted LLM call log
```

No result directories are included in this appendix.
