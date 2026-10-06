# SEAM: Self-Evolving Causal Memory

Code supplement for SEAM, including causal memory units, evidence verification,
memory evolution, baseline agents, evaluation drivers, and dataset helpers.
SEAM starts each evaluation run with empty memory and accumulates memory across
episodes within that run.

## Repository layout

```text
configs/                  Local inference config and API field template
experiments/              Sequential evaluation and ALFWorld ablations
scripts/                  Dataset checks and TextWorld generation scripts
src/agent/                SEAM agent and baseline implementations
src/env/                  ALFWorld, TextWorld, ScienceWorld, and WebShop interfaces
src/memory/               Causal memory units, graph, verification, and evolution
src/llm/                  OpenAI-compatible chat completion client
src/utils/                Experiment logging
```

The agent is exported as `SEAMAgent`; its implementation class remains named
`SECMAgent` for compatibility. Use `seam` in experiment commands.

## Installation

The SEAM code requires **Python 3.10 or newer**. Run all commands below from the
repository root.

```bash
git clone https://github.com/adtureven/SEAM.git
cd SEAM
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Install benchmark dependencies for the environment you plan to run:

```bash
python -m pip install alfworld      # ALFWorld data tools
python -m pip install textworld     # TextWorld generation and execution
python -m pip install scienceworld # ScienceWorld execution
```

`requirements-optional.txt` lists these optional packages and Gym. WebShop has
its own dependencies and setup procedure, described below.

## Model configuration

Use [configs/default.yaml](configs/default.yaml) as the complete experiment
configuration. It points to a local OpenAI-compatible endpoint at
`http://127.0.0.1:8000/v1`, with model name `qwen3-4b`.

For example, with vLLM installed separately and model weights available:

```bash
vllm serve /path/to/model --served-model-name qwen3-4b --port 8000
```

Edit `llm.base_url` and `llm.model` if your endpoint or served model name differs.

For an external API, replace the **entire `llm` section** in `configs/default.yaml`
with the fields in [configs/api_template.yaml](configs/api_template.yaml), then
set your provider's base URL and model ID. The template contains only LLM fields
and cannot be used as a complete experiment config. Replacing the section also
removes the local model's `extra_body.chat_template_kwargs`, which some APIs
do not accept.

```bash
export SEAM_API_KEY="your-api-key"
```

The client reads the key from this environment variable.

## Dataset preparation

`scripts/prepare_datasets.py` checks local files and dependencies and prints
setup guidance.

### ALFWorld

Obtain the `json_2.1.1` benchmark data and place it at
`data/alfworld/json_2.1.1`, or set `env.data_path` to your existing data directory.
The default split is `valid_unseen`. Each trial requires a solvable
`game.tw-pddl` file.

```bash
python scripts/prepare_datasets.py --dataset alfworld
```

The evaluation driver uses the included lightweight PDDL simulator in
`src/env/alfworld_pddl.py` to execute these tasks.

### TextWorld

With TextWorld installed, generate the Cooking and Treasure Hunter suites:

```bash
bash scripts/generate_textworld_cooking_games.sh
bash scripts/generate_textworld_treasure_games.sh
```

Each script generates 100 games per difficulty (easy, medium, hard), using fixed
seeds, and writes a `manifest.tsv`. Cooking games go to
`data/textworld_cooking/games`; Treasure Hunter games go to
`data/textworld_treasure/games`. The curriculum evaluations visit easy, then
medium, then hard games. The corresponding `_interleaved` environment names
alternate difficulty levels.

### ScienceWorld

Set `scienceworld.task_indices_file` to the ordered index list for the
270-instance subset. The default path is `data/scienceworld_270_indices.json`.
The loader accepts a JSON object with a `task_indices` list of integers.

Indices refer to the task list built by `src/env/sciworld_env.py`. With
`scienceworld.expand_variations: true`, this list enumerates all variations of
each task; with `false` (the current default), it contains one variation per
task. Set this field and `variation_idx` to match the supplied index list.
`scienceworld.split` is recorded as a label and does not select official
train/test variation partitions.

```bash
python scripts/prepare_datasets.py --dataset scienceworld
```

### Optional few-shot prompts

To use few-shot examples, place the prompt dictionary at
`data/react_prompts.json`.

### WebShop: all products, official 500-task test split

Clone the [upstream WebShop repository](https://github.com/princeton-nlp/WebShop)
beside this checkout:

```bash
git clone https://github.com/princeton-nlp/WebShop.git ../WebShop
```

Follow the [upstream setup instructions](https://github.com/princeton-nlp/WebShop#-setup)
with `setup.sh -d all` to install dependencies, download the full dataset, and
build the full search index. The setup script uses Conda and Java.

In the WebShop checkout, update `web_agent_site/utils.py` to read the **full
data files**, as required by upstream:

```python
DEFAULT_ATTR_PATH = join(BASE_DIR, '../data/items_ins_v2.json')
DEFAULT_FILE_PATH = join(BASE_DIR, '../data/items_shuffle.json')
```

This step matters: `num_products: null` removes the product count cap but still
loads the upstream `DEFAULT_FILE_PATH`. Leaving it at `items_shuffle_1000.json`
does not load the full catalogue. The full search index is
`WebShop/search_engine/indexes`.

Set these values in `configs/default.yaml` (only `repo_path` needs changing
from the shipped settings):

```yaml
webshop:
  repo_path: "../WebShop"
  split: "test"
  num_episodes: 500
  observation_mode: "text"
  num_products: null
  limit_goals: -1
  human_goals: true
  show_attrs: false
  max_steps: 15
```

The official evaluation uses human goals, as configured in the
[upstream baseline](https://github.com/princeton-nlp/WebShop/blob/master/baseline_models/train_rl.py).
The [official test split](https://github.com/princeton-nlp/WebShop/blob/master/baseline_models/env.py)
is goal indices 0–499 after the
[environment's fixed shuffle](https://github.com/princeton-nlp/WebShop/blob/master/web_agent_site/envs/web_agent_text_env.py)
(seed 233). The wrapper selects these 500 tasks and evaluates them in index order.
Keep `limit_goals: -1` and `human_goals: true` to preserve this split.

Each episode permits at most 15 environment actions. The wrapper records reward
on a 0–1 scale and counts success when the environment ends with reward at least
0.99. Memory accumulates across these test episodes.

## Evaluation

The sequential driver takes positional arguments in this order:

```text
python experiments/run_sequential.py NUM_EPISODES METHOD MAX_STEPS CONFIG --env ENVIRONMENT
```

All examples explicitly pass `configs/default.yaml`. A positive episode count
or step limit on the command line overrides the config. For WebShop, passing
`0` for both uses `webshop.num_episodes` and `webshop.max_steps`.

ALFWorld:

```bash
python experiments/run_sequential.py 134 seam 50 configs/default.yaml --env alfworld
```

ScienceWorld:

```bash
python experiments/run_sequential.py 270 seam 100 configs/default.yaml --env scienceworld
```

WebShop:

```bash
python experiments/run_sequential.py 500 seam 15 configs/default.yaml --env webshop
```

TextWorld Cooking and Treasure Hunter curricula:

```bash
python experiments/run_sequential.py 300 seam 50 configs/default.yaml --env textworld_cooking_curriculum
python experiments/run_sequential.py 300 seam 50 configs/default.yaml --env textworld_treasure_curriculum
```

For a baseline, replace `seam` with `baseline`, `react`, `reflexion`, `amem`,
`mem0`, `expel`, `awm`, `synapse`, `swiftsage`, `rag`, `autoguide`, or `rap`.
Use `baselines` to run all registered methods sequentially. These are the
baseline implementations included in this supplement.

## ALFWorld ablations

Run all five ablations or one selected variant:

```bash
python experiments/run_ablation.py all 134 50 configs/default.yaml
python experiments/run_ablation.py no_effect_prediction 134 50 configs/default.yaml
```

Available variants are `no_self_evolution`, `no_evidence_verification`,
`no_causal_abstraction`, `no_memory_revision`, and `no_effect_prediction`.
This driver supports ALFWorld only.

## Outputs

Runs create a timestamped directory under `experiment.log_dir` (`outputs/` by
default):

```text
config.yaml               Config snapshot with credentials and endpoint redacted
metadata.json             Run metadata
summary.json              Aggregate metrics and per-task results
episodes/                 Per-episode JSONL traces
memory/                   SEAM graph/evolution snapshots or baseline memory
prompts/                  Recorded prompts where saved by the driver
llm_calls.jsonl           LLM responses, usage, and shortened user input
```

`llm_calls.jsonl` redacts the endpoint but retains task text and model responses.
