#!/bin/bash
# =============================================================================
# check_setup.sh — Fast setup + pipeline check using the debug partition
#
# Uses the debug partition for near-instant allocation (no GPU, CPU only).
# Runs --dry-run to verify the full pipeline (imports, data loading,
# preprocessing, file writing) WITHOUT triggering model inference.
#
# Usage:
#   bash scripts/check_setup.sh
# =============================================================================

PROJECT_DIR=$(pwd)

salloc \
    --partition=debug \
    --cpus-per-task=4 \
    --mem=16G \
    --time=00:15:00 \
    --account=snazaria_1817 \
    bash -c "
        echo '========================================'
        echo ' Binoculars Setup Check'
        echo '========================================'
        echo ''

        # --- 1. Conda env ---
        echo '[1/4] Activating conda environment...'
        module purge
        module load conda
        conda run -n binoculars python -c 'import sys; print(\"  Python:\", sys.executable)' \
            && echo '  OK' || { echo '  FAIL — run bash scripts/setup_env.sh first'; exit 1; }

        # --- 2. Imports ---
        echo ''
        echo '[2/4] Checking all imports...'
        conda run -n binoculars python -c '
import datasets; print(\"  datasets  :\", datasets.__version__)
import transformers; print(\"  transformers:\", transformers.__version__)
import torch; print(\"  torch     :\", torch.__version__)
import pandas; print(\"  pandas    :\", pandas.__version__)
import numpy; print(\"  numpy     :\", numpy.__version__)
print(\"  OK\")
' || { echo '  FAIL — missing packages'; exit 1; }

        # --- 3. Binoculars library ---
        echo ''
        echo '[3/4] Checking Binoculars library...'
        conda run -n binoculars python -c '
from binoculars import Binoculars
print(\"  Binoculars class found: OK\")
' || { echo '  FAIL — install with: pip install git+https://github.com/ahans30/Binoculars.git'; exit 1; }

        # --- 4. Dry-run pipeline ---
        echo ''
        echo '[4/4] Running dry-run pipeline (5 rows, no inference)...'
        conda run -n binoculars python $PROJECT_DIR/main.py \
            --source hf \
            --sample 5 \
            --dry-run \
            --output check_setup_dryrun.json \
            && echo '  Pipeline OK' || { echo '  FAIL — check error above'; exit 1; }

        echo ''
        echo '========================================'
        echo ' All checks passed — ready to run jobs!'
        echo '========================================'
    "
