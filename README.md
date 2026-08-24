# cpo_trl

Codebase for CPO experiments with TRL-style baselines. The repository is organized around three separate concerns:

- reusable method code in `src/cpo_trl/`
- experiment configuration in `configs/`
- run organization and generated artifacts in `experiments/`, `outputs/`, and `analysis/`

## Important Training Assumption

DPO, KTO, and CPO start directly from instruction-tuned base models such as:

```text
Qwen/Qwen2.5-1.5B-Instruct
```

They do not use an intermediate SFT checkpoint as the starting model. SFT remains available as an optional baseline or diagnostic experiment, but it is not part of the default training chain.

## Layout

```text
configs/       Shared, method, and ablation configs
src/cpo_trl/   Reusable CPO, data, sampling, evaluation, and trainer code
scripts/       Thin entrypoints for data, training, evaluation, audits, experiments
experiments/   Scientific run organization by research question
outputs/       Large generated artifacts: checkpoints, generations, judge outputs, logs
analysis/      Paper-ready tables, figures, and notebooks
tests/         Unit, integration, and regression tests
```

The internal Python package is split by responsibility:

```text
src/cpo_trl/data/          JSONL schemas and prompt formatting
src/cpo_trl/losses/        CPO, unary CPO, DPO, and KTO loss entrypoints
src/cpo_trl/references/    EMA and log-ratio/reference helpers
src/cpo_trl/sampling/      Pair-aware and sampling utilities
src/cpo_trl/trainers/      Stateful trainer/loss-computer components
src/cpo_trl/evaluation/    Teacher-forced evaluation utilities
src/cpo_trl/metrics/       Preference metrics callbacks and records
src/cpo_trl/models/        Model loading and PEFT helpers
src/cpo_trl/utils/         Finite checks and compatibility utilities
```

Compatibility wrappers such as `cpo_trl.losses`, `cpo_trl.data`, and
`cpo_trl.cpo_trainer` remain available while scripts and tests migrate to the
new explicit paths.

## Install

```bash
poetry install
```

or, without Poetry:

```bash
pip install -r requirements.txt
```

## Prepare Data

```bash
poetry run python scripts/data/prepare_ultrafeedback.py --limit 10000
```

The target processed layout is:

```text
data/processed/sft/train.jsonl
data/processed/dpo/train.jsonl
data/processed/kto/train.jsonl
data/processed/cpo/train.jsonl
```

Use the same prompt IDs across methods and keep evaluation IDs in `data/manifests/`.

## Train Baselines and CPO

DPO:

```bash
poetry run python scripts/train/train_dpo.py --config configs/dpo/dpo_controlled.yaml
```

KTO:

```bash
poetry run python scripts/train/train_kto.py --config configs/kto/kto_controlled.yaml
```

CPO:

```bash
poetry run python scripts/train/train_cpo.py --config configs/cpo/cpo_controlled.yaml
```

CPO unary ablation:

```bash
poetry run python scripts/train/train_cpo.py --config configs/cpo/cpo_unary.yaml
```

All default method configs use the Instruct model directly through `model_name_or_path` and write checkpoints under `outputs/checkpoints/`.

## Evaluation

Keep generation and judging separate.

Generate model responses first:

```bash
poetry run python scripts/data/build_manifests.py \
  --source-file data/processed/dpo/validation.jsonl

poetry run python scripts/evaluate/generate_from_prompts.py \
  --prompts-file data/manifests/eval_manifest_natural.jsonl \
  --models DPO=outputs/checkpoints/dpo KTO=outputs/checkpoints/kto CPO=outputs/checkpoints/cpo \
  --temperature 0 \
  --seed 42 \
  --output-jsonl outputs/generations/main.jsonl
```

Then run the judge over saved generations. The judge does not need training
checkpoints in this mode:

```bash
poetry run python scripts/evaluate/evaluate_judge.py \
  --generations-file outputs/generations/main.jsonl \
  --position-balanced \
  --output-jsonl outputs/judge/main_pairwise.jsonl \
  --summary-json outputs/judge/main_summary.json

poetry run python scripts/evaluate/evaluate_winrate.py --help
```

## Run Manifests

Each experiment run should include a `run_manifest.json` with at least:

```json
{
  "experiment": "E4_alpha_sweep",
  "method": "cpo",
  "seed": 42,
  "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
  "alpha": 0.25,
  "beta": 0.1,
  "reference_type": "token_kl",
  "sampler": "pair_aware",
  "checkpoint_path": "outputs/checkpoints/cpo/E4_alpha_025_seed42"
}
```

## Tests

```bash
poetry run pytest
```
