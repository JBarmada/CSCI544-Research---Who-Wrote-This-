#!/bin/bash

set -euo pipefail

PINNED_COMMIT="c8ae2f90d50ee696418bc71d8d9e5020e5f9d7b8"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPSTREAM_DIR="${UPSTREAM_DIR:-$PROJECT_DIR/.upstream/Binoculars}"
PYTHON="${PYTHON:-/home1/barmada/.conda/envs/binoculars/bin/python}"
DATASET_PATH="${DATASET_PATH:-$UPSTREAM_DIR/datasets/core/cc_news/cc_news-llama2_13.jsonl}"
HUMAN_SAMPLE_KEY="${HUMAN_SAMPLE_KEY:-text}"
MACHINE_SAMPLE_KEY="${MACHINE_SAMPLE_KEY:-meta-llama-Llama-2-13b-hf_generated_text_wo_prompt}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
unset PYTHONPATH
unset PYTHONHOME

salloc \
    --partition=debug \
    --cpus-per-task=4 \
    --mem=16G \
    --time=00:15:00 \
    --account=snazaria_1817 \
    bash -lc "
        set -euo pipefail
        module purge >/dev/null 2>&1 || true
        export OPENBLAS_NUM_THREADS='$OPENBLAS_NUM_THREADS'
        export OMP_NUM_THREADS='$OMP_NUM_THREADS'
        export MKL_NUM_THREADS='$MKL_NUM_THREADS'
        export NUMEXPR_NUM_THREADS='$NUMEXPR_NUM_THREADS'
        export TOKENIZERS_PARALLELISM='$TOKENIZERS_PARALLELISM'
        export PYTHONNOUSERSITE='$PYTHONNOUSERSITE'
        unset PYTHONPATH
        unset PYTHONHOME
        echo '========================================'
        echo ' Binoculars Reproduction Check'
        echo '========================================'
        echo

        echo '[1/4] Python executable'
        '$PYTHON' -c 'import sys; print(sys.executable)'
        echo

        echo '[2/4] Import check'
        '$PYTHON' -c 'from binoculars import Binoculars; import datasets; import matplotlib; import sklearn; print(\"imports ok\")'
        echo

        echo '[3/4] Upstream pin check'
        git -C '$UPSTREAM_DIR' rev-parse HEAD
        test \"\$(git -C '$UPSTREAM_DIR' rev-parse HEAD)\" = '$PINNED_COMMIT'
        echo

        echo '[4/4] Dataset wiring check'
        '$PYTHON' -c \"from datasets import Dataset; ds = Dataset.from_json(r'$DATASET_PATH'); ds = ds.select(range(min(2, len(ds)))); assert '$HUMAN_SAMPLE_KEY' in ds.column_names; assert '$MACHINE_SAMPLE_KEY' in ds.column_names; print(ds.column_names)\"
        echo
        echo 'Setup looks ready.'
    "
