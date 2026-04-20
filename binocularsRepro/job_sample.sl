#!/bin/bash
#SBATCH --job-name=bino-sample
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/sample_%j.out
#SBATCH --error=logs/sample_%j.err
#SBATCH --account=snazaria_1817

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UPSTREAM_DIR="${UPSTREAM_DIR:-$PROJECT_DIR/.upstream/Binoculars}"
PYTHON="${PYTHON:-/home1/barmada/.conda/envs/binoculars/bin/python}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
unset PYTHONPATH
unset PYTHONHOME

LIMIT="${LIMIT:-64}"
DATASET_PATH="${DATASET_PATH:-$UPSTREAM_DIR/datasets/core/cc_news/cc_news-llama2_13.jsonl}"
DATASET_NAME="${DATASET_NAME:-CC-News}"
HUMAN_SAMPLE_KEY="${HUMAN_SAMPLE_KEY:-text}"
MACHINE_SAMPLE_KEY="${MACHINE_SAMPLE_KEY:-meta-llama-Llama-2-13b-hf_generated_text_wo_prompt}"
MACHINE_TEXT_SOURCE="${MACHINE_TEXT_SOURCE:-LLaMA-2-13B}"
TOKENS_SEEN="${TOKENS_SEEN:-512}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MODE="${MODE:-accuracy}"
JOB_NAME="${JOB_NAME:-sample-${DATASET_NAME}-${LIMIT}}"

mkdir -p "$PROJECT_DIR/logs" "$PROJECT_DIR/results"
cd "$PROJECT_DIR"

echo "Job ID             : $SLURM_JOB_ID"
echo "Sample limit       : $LIMIT"
echo "Dataset path       : $DATASET_PATH"
echo "Started at         : $(date)"
echo "----------------------------------------"

$PYTHON "$PROJECT_DIR/main.py" \
    --dataset_path "$DATASET_PATH" \
    --dataset_name "$DATASET_NAME" \
    --human_sample_key "$HUMAN_SAMPLE_KEY" \
    --machine_sample_key "$MACHINE_SAMPLE_KEY" \
    --machine_text_source "$MACHINE_TEXT_SOURCE" \
    --tokens_seen "$TOKENS_SEEN" \
    --batch_size "$BATCH_SIZE" \
    --mode "$MODE" \
    --job_name "$JOB_NAME" \
    --limit "$LIMIT"

echo "----------------------------------------"
echo "Finished at        : $(date)"
