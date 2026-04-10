#!/bin/bash
# =============================================================================
# check_setup.sh — Fast setup + pipeline check using the debug partition
#
# Bypasses module load and conda run entirely by calling the env's Python
# directly — much faster startup.
#
# Usage:
#   bash scripts/check_setup.sh
# =============================================================================

PROJECT_DIR=$(pwd)
PYTHON="/home1/barmada/.conda/envs/binoculars/bin/python"

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

        # --- 1. Python env ---
        echo '[1/4] Python executable...'
        $PYTHON -c 'import sys; print(\"  \", sys.executable)' \
            || { echo '  FAIL — env not found at $PYTHON'; exit 1; }

        # --- 2+3. All imports + Binoculars in one shot ---
        echo ''
        echo '[2/4] Checking all imports...'
        $PYTHON -c '
import sys

checks = [
    (\"datasets\",     \"datasets\"),
    (\"transformers\", \"transformers\"),
    (\"torch\",        \"torch\"),
    (\"pandas\",       \"pandas\"),
    (\"numpy\",        \"numpy\"),
]

all_ok = True
for label, mod in checks:
    try:
        m = __import__(mod)
        ver = getattr(m, \"__version__\", \"?\")
        print(f\"  {label:<14} {ver}\")
    except ImportError as e:
        print(f\"  {label:<14} MISSING — {e}\")
        all_ok = False

if not all_ok:
    sys.exit(1)
print(\"  OK\")
' || exit 1

        echo ''
        echo '[3/4] Checking Binoculars library...'
        $PYTHON -c '
from binoculars import Binoculars
print(\"  Binoculars: OK\")
' || { echo '  FAIL — run: pip install git+https://github.com/ahans30/Binoculars.git'; exit 1; }

        # --- 4. Dry-run pipeline ---
        echo ''
        echo '[4/4] Dry-run pipeline (5 rows, no inference)...'
        $PYTHON $PROJECT_DIR/main.py \
            --source hf \
            --sample 5 \
            --dry-run \
            --output check_setup_dryrun.json \
            && echo '  Pipeline: OK' || exit 1

        echo ''
        echo '========================================'
        echo ' All checks passed — ready to run jobs!'
        echo '========================================'
    "
