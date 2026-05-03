#!/bin/bash

set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <output-file> <python-executable>"
    exit 1
fi

OUTPUT_FILE="$1"
PYTHON_BIN="$2"
OUTPUT_DIR="$(dirname "$OUTPUT_FILE")"

mkdir -p "$OUTPUT_DIR"

run_if_available() {
    local label="$1"
    shift
    echo "===== $label =====" >> "$OUTPUT_FILE"
    if command -v "$1" >/dev/null 2>&1; then
        "$@" >> "$OUTPUT_FILE" 2>&1 || true
    else
        echo "Command not available: $1" >> "$OUTPUT_FILE"
    fi
    echo >> "$OUTPUT_FILE"
}

echo "===== Run Metadata =====" > "$OUTPUT_FILE"
echo "Timestamp: $(date -Is)" >> "$OUTPUT_FILE"
echo "Hostname: $(hostname)" >> "$OUTPUT_FILE"
echo "Working directory: $(pwd)" >> "$OUTPUT_FILE"
echo "User: ${USER:-unknown}" >> "$OUTPUT_FILE"
echo >> "$OUTPUT_FILE"

echo "===== SLURM Environment =====" >> "$OUTPUT_FILE"
env | grep '^SLURM_' | sort >> "$OUTPUT_FILE" 2>/dev/null || true
echo >> "$OUTPUT_FILE"

echo "===== Thread / Python Environment =====" >> "$OUTPUT_FILE"
echo "OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-}" >> "$OUTPUT_FILE"
echo "OMP_NUM_THREADS=${OMP_NUM_THREADS:-}" >> "$OUTPUT_FILE"
echo "MKL_NUM_THREADS=${MKL_NUM_THREADS:-}" >> "$OUTPUT_FILE"
echo "NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-}" >> "$OUTPUT_FILE"
echo "TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-}" >> "$OUTPUT_FILE"
echo "PYTHONNOUSERSITE=${PYTHONNOUSERSITE:-}" >> "$OUTPUT_FILE"
echo "PYTHONPATH=${PYTHONPATH:-}" >> "$OUTPUT_FILE"
echo "PYTHONHOME=${PYTHONHOME:-}" >> "$OUTPUT_FILE"
echo >> "$OUTPUT_FILE"

run_if_available "uname -a" uname -a
run_if_available "lscpu" lscpu
run_if_available "free -h" free -h
run_if_available "nvidia-smi -L" nvidia-smi -L
run_if_available "nvidia-smi" nvidia-smi

echo "===== Python Runtime =====" >> "$OUTPUT_FILE"
"$PYTHON_BIN" --version >> "$OUTPUT_FILE" 2>&1 || true
"$PYTHON_BIN" - <<'PY' >> "$OUTPUT_FILE" 2>&1 || true
import inspect
import os
import platform
import sys

print(f"sys.executable: {sys.executable}")
print(f"sys.version: {sys.version}")
print(f"platform: {platform.platform()}")
print(f"cwd: {os.getcwd()}")

try:
    import torch
    print(f"torch: {torch.__version__}")
    print(f"torch.cuda.is_available: {torch.cuda.is_available()}")
    print(f"torch.cuda.device_count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        for idx in range(torch.cuda.device_count()):
            print(f"torch.cuda.get_device_name({idx}): {torch.cuda.get_device_name(idx)}")
except Exception as exc:
    print(f"torch import failed: {exc}")

for mod_name in ["numpy", "datasets", "transformers", "sklearn", "matplotlib", "binoculars"]:
    try:
        mod = __import__(mod_name)
        print(f"{mod_name}: version={getattr(mod, '__version__', 'unknown')} file={getattr(mod, '__file__', 'unknown')}")
    except Exception as exc:
        print(f"{mod_name} import failed: {exc}")

try:
    from binoculars import Binoculars
    print(f"Binoculars class source: {inspect.getsourcefile(Binoculars)}")
except Exception as exc:
    print(f"Binoculars source lookup failed: {exc}")
PY
echo >> "$OUTPUT_FILE"

run_if_available "pip show Binoculars" "$PYTHON_BIN" -m pip show Binoculars
run_if_available "pip show numpy" "$PYTHON_BIN" -m pip show numpy
run_if_available "pip show torch" "$PYTHON_BIN" -m pip show torch
run_if_available "pip show transformers" "$PYTHON_BIN" -m pip show transformers
