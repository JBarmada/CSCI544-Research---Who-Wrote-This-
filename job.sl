#!/bin/bash
#SBATCH --job-name=bino-full
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/full_%j.out
#SBATCH --error=logs/full_%j.err

#SBATCH --account=snazaria_1817

# =============================================================================
# job.sl — sbatch job for full dataset runs
#
# Submit with:
#   sbatch job.sl                           # full HF dataset, accuracy mode
#   MODE=low-fpr sbatch job.sl              # low-fpr threshold
#   OUTPUT=my_results.csv sbatch job.sl     # custom output name
#
# GPU note: Two Falcon-7B models in bfloat16 ≈ 28 GB VRAM total.
#   - a100:1 (80 GB) — comfortable, recommended
#   - If only V100 (32 GB) nodes are available, change gres to:
#       #SBATCH --gres=gpu:v100:2
#     The Binoculars class uses device_map="auto" and will split across GPUs.
# =============================================================================

module purge
module load conda
conda activate binoculars

MODE=${MODE:-accuracy}
OUTPUT=${OUTPUT:-full_results.csv}

echo "Job ID       : $SLURM_JOB_ID"
echo "Node         : $SLURMD_NODENAME"
echo "Mode         : $MODE"
echo "Output file  : $OUTPUT"
echo "Started at   : $(date)"
echo "----------------------------------------"

python main.py \
    --source hf \
    --mode "$MODE" \
    --output "$OUTPUT"

echo "----------------------------------------"
echo "Finished at  : $(date)"
