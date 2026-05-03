# Binoculars CARC Reproduction

This directory is a CARC-focused thin wrapper around the official `ahans30/Binoculars` release, pinned to commit `c8ae2f90d50ee696418bc71d8d9e5020e5f9d7b8`.

It is set up to reproduce the paper-style core runs first on the shipped upstream datasets:

- CC-News
- CNN
- PubMed

## Quick Start

If the repo is already on CARC:

```bash
cd ~/CSCI544-Research---Who-Wrote-This-/binocularsRepro
git pull
bash scripts/setup_env.sh
bash scripts/check_setup.sh
sbatch job_sample.sl
sbatch job.sl
```

If you need to clone the repo first:

```bash
git clone git@github.com:JBarmada/CSCI544-Research---Who-Wrote-This-.git
cd CSCI544-Research---Who-Wrote-This-/binocularsRepro
bash scripts/setup_env.sh
bash scripts/check_setup.sh
sbatch job_sample.sl
sbatch job.sl
```

Run all three paper-core datasets:

```bash
cd ~/CSCI544-Research---Who-Wrote-This-/binocularsRepro
git pull
bash scripts/setup_env.sh
bash scripts/submit_core.sh
```

Analyze a finished run:

```bash
${BINO_PYTHON:-$(which python)} analyze.py --input results/<job_name>
```

Run a CC-News diagnostic that stops after model load:

```bash
STOP_AFTER=model_load sbatch --mem=64G job_diagnostic.sl
```

Run the same diagnostic at higher memory:

```bash
STOP_AFTER=model_load sbatch --mem=128G job_diagnostic.sl
STOP_AFTER=model_load sbatch --mem=192G job_diagnostic.sl
```

Run later stop-points one at a time:

```bash
STOP_AFTER=dataset_load sbatch --mem=128G job_diagnostic.sl
STOP_AFTER=human_pass sbatch --mem=192G job_diagnostic.sl
```

Watch queue and logs:

```bash
squeue -u $USER
tail -f logs/sample_<jobid>.out
tail -f logs/full_<jobid>.out
```

## What Each Command Does

### `bash scripts/setup_env.sh`

Purpose:

- clones the official upstream Binoculars repo into `.upstream/Binoculars`
- checks out the pinned commit
- creates or reuses the `binoculars` conda env
- installs the upstream package from that local clone
- installs local dependencies for plotting and analysis
- applies CARC-safe runtime defaults such as `numpy<2`

Use it when:

- this is your first run on CARC
- you pulled changes to `binocularsRepro`
- you want to rebuild the environment after package issues

### `bash scripts/check_setup.sh`

Purpose:

- requests a short debug allocation with `salloc`
- verifies the Python executable path
- checks imports for `binoculars`, `datasets`, `matplotlib`, and `sklearn`
- verifies the upstream clone is pinned to the expected commit
- verifies the default CC-News dataset path and required column names

Use it when:

- you want to validate the environment before spending a GPU job
- you changed the setup scripts or package versions

### `sbatch job_sample.sl`

Purpose:

- submits a small smoke-test GPU job
- uses the same methodology as the full job
- defaults to a limited sample on CC-News

Default behavior:

- partition: `gpu`
- gpu: `a40:1`
- CPUs: `8`
- memory: `32G`
- time: `1:00:00`
- sample limit: `64`
- mode: `accuracy`

Use it when:

- you want to confirm the pipeline runs end-to-end
- you want output artifacts quickly before launching the full run

### `sbatch job.sl`

Purpose:

- submits the full CARC job for one dataset configuration
- runs the paper-style Binoculars experiment flow
- scores human and machine text separately
- saves metrics and artifacts into one run directory

Default behavior:

- dataset: CC-News
- human key: `text`
- machine key: `meta-llama-Llama-2-13b-hf_generated_text_wo_prompt`
- machine source: `LLaMA-2-13B`
- tokens seen: `512`
- batch size: `32`
- mode: `accuracy`
- partition: `gpu`
- gpu: `a40:1`
- CPUs: `16`
- memory: `64G`
- time: `24:00:00`

Use it when:

- you want the default paper-style full run
- you want one custom run by overriding env vars at submit time

### `sbatch job_diagnostic.sl`

Purpose:

- submits a diagnostic GPU job without changing the released detector logic
- records memory checkpoints before and after the major stages
- can stop after model load, dataset load, or the human scoring pass

Default behavior:

- dataset: CC-News
- stop point: `model_load`
- limit: `64`
- partition: `gpu`
- gpu: `a40:1`
- CPUs: `16`
- memory: `64G`
- time: `1:00:00`

Use it when:

- you want to isolate where OOM occurs
- you want one-at-a-time CC-News memory profiling
- you want `experiments_details.json` and `diagnostic_status.json` even for partial runs

### `bash scripts/submit_core.sh`

Purpose:

- submits the three core paper-style runs:
  - CC-News
  - CNN
  - PubMed

Use it when:

- you want the main benchmark runs without typing each submit command manually

### `python analyze.py --input results/<job_name>`

Purpose:

- reads a finished run directory
- loads `score_df.csv` and `experiments_details.json`
- writes `analysis.json`
- summarizes overall score statistics, class-wise score statistics, and prediction counts

Use it when:

- you want a cleaner post-run summary for reporting or later comparisons

## Custom Submit Examples

Run CNN manually:

```bash
DATASET_PATH=$PWD/.upstream/Binoculars/datasets/core/cnn/cnn-llama2_13.jsonl \
DATASET_NAME=CNN \
HUMAN_SAMPLE_KEY=article \
MACHINE_SAMPLE_KEY=meta-llama-Llama-2-13b-hf_generated_text_wo_prompt \
MACHINE_TEXT_SOURCE=LLaMA-2-13B \
sbatch job.sl
```

Run PubMed manually:

```bash
DATASET_PATH=$PWD/.upstream/Binoculars/datasets/core/pubmed/pubmed-llama2_13.jsonl \
DATASET_NAME=PubMed \
HUMAN_SAMPLE_KEY=article \
MACHINE_SAMPLE_KEY=meta-llama-Llama-2-13b-hf_generated_text_wo_prompt \
MACHINE_TEXT_SOURCE=LLaMA-2-13B \
sbatch job.sl
```

Run low-FPR mode:

```bash
MODE=low-fpr sbatch job.sl
```

Run a different sample size:

```bash
LIMIT=128 sbatch job_sample.sl
```

Set a custom run directory name:

```bash
JOB_NAME=my_ccnews_run sbatch job.sl
```

## Methodology Defaults

The wrapper preserves these methodology choices from the upstream release:

- observer model: `tiiuae/falcon-7b`
- performer model: `tiiuae/falcon-7b-instruct`
- token limit: `512`
- default threshold mode: `accuracy`
- accuracy threshold: `0.9015310749276843`
- low-FPR threshold: `0.8536432310785527`
- no extra preprocessing or word-count filtering
- separate human and machine scoring before concatenation

## Output Files

Each completed run writes to `results/<job_name>/`:

- `score_df.csv`
  - row-level score table with `score`, `class`, and `pred`
- `performance.png`
  - ROC plot with F1, ROC-AUC, and TPR at `0.01%` FPR
- `experiments_details.json`
  - run metadata, thresholds, dataset settings, models, and summary metrics
- `system_info.txt`
  - methodology-grade system snapshot including SLURM vars, `lscpu`, `free -h`, `nvidia-smi`, Python package provenance, and Binoculars import path
- `analysis.json`
  - produced only after running `analyze.py`

Diagnostic runs also write:

- `diagnostic_status.json`
  - last successful checkpoint, current status, and latest memory snapshot
- `experiments_details.json`
  - includes `memory_checkpoints` for:
    - process start
    - before/after model load
    - after dataset load
    - after optional limit
    - before/after human scoring
    - before/after machine scoring
    - before final save

## Troubleshooting

If setup or import checks fail, rerun:

```bash
cd ~/CSCI544-Research---Who-Wrote-This-/binocularsRepro
git pull
bash scripts/setup_env.sh
bash scripts/check_setup.sh
```

If you need to inspect the installed package source directly:

```bash
git -C .upstream/Binoculars rev-parse HEAD
${BINO_PYTHON:-$(which python)} -c "import binoculars; print(binoculars.__file__)"
```
