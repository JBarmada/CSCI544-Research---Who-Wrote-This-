# Binoculars CARC Reproduction

This directory is a CARC-focused thin wrapper around the official `ahans30/Binoculars` release, pinned to commit `c8ae2f90d50ee696418bc71d8d9e5020e5f9d7b8`.

It is designed to reproduce the paper-style core runs first on the shipped upstream datasets:

- CC-News
- CNN
- PubMed

## Setup

Run on CARC from this directory:

```bash
bash scripts/setup_env.sh
```

That script:

- clones the upstream repo into `.upstream/Binoculars`
- checks out the pinned commit
- creates a `binoculars` conda env
- installs the upstream package and local analysis dependencies

## Validate wiring

```bash
bash scripts/check_setup.sh
```

This verifies:

- the upstream clone exists
- the commit is pinned correctly
- required imports work
- the default CC-News dataset path and keys resolve

## Run jobs

Default full run:

```bash
sbatch job.sl
```

Quick smoke/sample run:

```bash
sbatch job_sample.sl
```

Submit the three paper-core datasets:

```bash
bash scripts/submit_core.sh
```

## Defaults

`job.sl` defaults to:

- dataset: CC-News
- human key: `text`
- machine key: `meta-llama-Llama-2-13b-hf_generated_text_wo_prompt`
- machine source: `LLaMA-2-13B`
- mode: `accuracy`
- tokens seen: `512`
- batch size: `32`

Override any of these with environment variables at submit time.

## Outputs

Each run writes to `results/<job_name>/`:

- `score_df.csv`
- `performance.png`
- `experiments_details.json`
- `analysis.json` after running `python analyze.py --input results/<job_name>`
