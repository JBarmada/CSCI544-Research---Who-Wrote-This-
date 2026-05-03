#!/bin/bash
#SBATCH --job-name=bino-repro
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/full_%j.out
#SBATCH --error=logs/full_%j.err
#SBATCH --account=your_carc_account

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
UPSTREAM_DIR="${UPSTREAM_DIR:-$PROJECT_DIR/.upstream/Binoculars}"
PYTHON="${PYTHON:-${BINO_PYTHON:-$(which python)}}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
unset PYTHONPATH
unset PYTHONHOME

DATASET_PATH="${DATASET_PATH:-$UPSTREAM_DIR/datasets/core/cc_news/cc_news-llama2_13.jsonl}"
DATASET_NAME="${DATASET_NAME:-CC-News}"
HUMAN_SAMPLE_KEY="${HUMAN_SAMPLE_KEY:-text}"
MACHINE_SAMPLE_KEY="${MACHINE_SAMPLE_KEY:-meta-llama-Llama-2-13b-hf_generated_text_wo_prompt}"
MACHINE_TEXT_SOURCE="${MACHINE_TEXT_SOURCE:-LLaMA-2-13B}"
TOKENS_SEEN="${TOKENS_SEEN:-512}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MODE="${MODE:-accuracy}"
DEFAULT_JOB_NAME="${DATASET_NAME}-${MACHINE_TEXT_SOURCE}-${TOKENS_SEEN}-tokens"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    DEFAULT_JOB_NAME="${DEFAULT_JOB_NAME}-job${SLURM_JOB_ID}"
fi
JOB_NAME="${JOB_NAME:-$DEFAULT_JOB_NAME}"
LIMIT_ARG=""
RUN_DIR="$PROJECT_DIR/results/$JOB_NAME"
SYSTEM_INFO_FILE="$RUN_DIR/system_info.txt"

if [[ -n "${LIMIT:-}" ]]; then
    LIMIT_ARG="--limit $LIMIT"
fi

mkdir -p "$PROJECT_DIR/logs" "$PROJECT_DIR/results" "$RUN_DIR"
cd "$PROJECT_DIR"

echo "Job ID             : $SLURM_JOB_ID"
echo "Node               : $SLURMD_NODENAME"
echo "Dataset path       : $DATASET_PATH"
echo "Human key          : $HUMAN_SAMPLE_KEY"
echo "Machine key        : $MACHINE_SAMPLE_KEY"
echo "Machine source     : $MACHINE_TEXT_SOURCE"
echo "Tokens seen        : $TOKENS_SEEN"
echo "Batch size         : $BATCH_SIZE"
echo "Mode               : $MODE"
echo "Job name           : $JOB_NAME"
echo "Started at         : $(date)"
echo "----------------------------------------"

bash "$PROJECT_DIR/scripts/save_system_info.sh" "$SYSTEM_INFO_FILE" "$PYTHON"

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
    $LIMIT_ARG

echo "----------------------------------------"
echo "Finished at        : $(date)"
