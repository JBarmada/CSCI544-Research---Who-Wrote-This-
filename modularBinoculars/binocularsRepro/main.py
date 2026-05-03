import argparse
import datetime
import json
import os
import resource
import socket
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from datasets import Dataset, logging as datasets_logging
from sklearn import metrics

from binoculars import detector as detector_module
from binoculars.detector import (
    BINOCULARS_ACCURACY_THRESHOLD,
    BINOCULARS_FPR_THRESHOLD,
    Binoculars,
)


UPSTREAM_COMMIT = "c8ae2f90d50ee696418bc71d8d9e5020e5f9d7b8"
OBSERVER_MODEL = "tiiuae/falcon-7b"
PERFORMER_MODEL = "tiiuae/falcon-7b-instruct"
STOP_AFTER_CHOICES = ["model_load", "dataset_load", "human_pass"]

COLOR = "black"
mpl.rcParams["text.color"] = COLOR
mpl.rcParams["axes.labelcolor"] = COLOR
mpl.rcParams["xtick.color"] = COLOR
mpl.rcParams["ytick.color"] = COLOR
mpl.rcParams["figure.dpi"] = 200
sns.set(style="darkgrid")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a CARC-friendly Binoculars reproduction on a JSONL dataset.",
    )
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the jsonl file")
    parser.add_argument("--dataset_name", type=str, default=None, help="Name of the dataset")
    parser.add_argument("--human_sample_key", type=str, required=True, help="Key for the human-generated text")
    parser.add_argument(
        "--machine_sample_key",
        type=str,
        default=None,
        help="Key for the machine-generated text",
    )
    parser.add_argument(
        "--machine_text_source",
        type=str,
        default=None,
        help="Name of model used to generate machine text",
    )
    parser.add_argument(
        "--tokens_seen",
        type=int,
        default=512,
        help="Number of tokens seen by the model",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for scoring")
    parser.add_argument("--job_name", type=str, default=None, help="Optional results directory name")
    parser.add_argument(
        "--mode",
        choices=["accuracy", "low-fpr"],
        default="accuracy",
        help="Binoculars threshold mode. Official experiments default to accuracy mode.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row cap for smoke tests.",
    )
    parser.add_argument(
        "--stop_after",
        choices=STOP_AFTER_CHOICES,
        default=None,
        help="Diagnostic stop-point. Writes checkpoint metadata and exits after the selected stage.",
    )
    return parser.parse_args()


def convert_to_pandas(human_scores, machine_scores):
    human_scores = human_scores["score"]
    machine_scores = machine_scores["score"]
    return pd.DataFrame(
        {"score": human_scores + machine_scores, "class": [0] * len(human_scores) + [1] * len(machine_scores)}
    )


def resolve_machine_sample_key(args, ds):
    if args.machine_sample_key:
        return args.machine_sample_key
    matches = [name for name in list(ds.features.keys())[::-1] if "generated_text" in name]
    if not matches:
        raise ValueError("No machine-generated column found. Pass --machine_sample_key explicitly.")
    return matches[0]


def resolve_machine_text_source(args):
    if args.machine_text_source:
        return args.machine_text_source
    suffix = "_generated_text_wo_prompt"
    if args.machine_sample_key and args.machine_sample_key.endswith(suffix):
        return args.machine_sample_key[: -len(suffix)]
    return args.machine_sample_key or "auto-machine-source"


def resolve_threshold(mode):
    if mode == "accuracy":
        return BINOCULARS_ACCURACY_THRESHOLD
    return BINOCULARS_FPR_THRESHOLD


def save_json(payload, save_path):
    with open(save_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=4)


def get_rss_mb():
    status_path = Path("/proc/self/status")
    if status_path.exists():
        with open(status_path, "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    rss_kb = int(line.split()[1])
                    return round(rss_kb / 1024, 2)

    max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if os.uname().sysname == "Darwin":
        return round(max_rss / (1024 * 1024), 2)
    return round(max_rss / 1024, 2)


def get_cuda_snapshot():
    snapshot = {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_cuda_available": torch.cuda.is_available(),
        "torch_cuda_device_count": torch.cuda.device_count(),
        "resolved_device_1": detector_module.DEVICE_1,
        "resolved_device_2": detector_module.DEVICE_2,
        "gpu_memory": [],
    }

    if torch.cuda.is_available():
        for idx in range(torch.cuda.device_count()):
            snapshot["gpu_memory"].append(
                {
                    "index": idx,
                    "name": torch.cuda.get_device_name(idx),
                    "memory_allocated_mb": round(torch.cuda.memory_allocated(idx) / (1024 * 1024), 2),
                    "memory_reserved_mb": round(torch.cuda.memory_reserved(idx) / (1024 * 1024), 2),
                }
            )
    return snapshot


def get_model_device(model):
    try:
        return str(next(model.parameters()).device)
    except Exception:
        return "unknown"


def build_base_metadata(args):
    return {
        "dataset_path": args.dataset_path,
        "dataset_name": args.dataset_name,
        "human_sample_key": args.human_sample_key,
        "machine_sample_key": args.machine_sample_key,
        "machine_text_source": args.machine_text_source,
        "tokens_seen": args.tokens_seen,
        "batch_size": args.batch_size,
        "job_name": args.job_name,
        "mode": args.mode,
        "threshold": args.threshold,
        "experiment_path": args.experiment_path,
        "observer_model": OBSERVER_MODEL,
        "performer_model": PERFORMER_MODEL,
        "upstream_commit": UPSTREAM_COMMIT,
        "start_time": args.start_time,
        "stop_after": args.stop_after,
        "last_successful_checkpoint": getattr(args, "last_successful_checkpoint", None),
        "status": getattr(args, "status", "running"),
        "memory_checkpoints": getattr(args, "memory_checkpoints", []),
    }


def persist_state(args, extra=None):
    payload = build_base_metadata(args)
    if extra:
        payload.update(extra)
    save_json(payload, os.path.join(args.experiment_path, "experiments_details.json"))

    diagnostic_payload = {
        "job_name": args.job_name,
        "status": payload["status"],
        "stop_after": args.stop_after,
        "last_successful_checkpoint": payload["last_successful_checkpoint"],
        "dataset_name": args.dataset_name,
        "dataset_path": args.dataset_path,
        "memory_checkpoints_count": len(payload["memory_checkpoints"]),
        "latest_checkpoint": payload["memory_checkpoints"][-1] if payload["memory_checkpoints"] else None,
        "updated_at": datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y"),
    }
    save_json(diagnostic_payload, os.path.join(args.experiment_path, "diagnostic_status.json"))


def record_checkpoint(args, checkpoint_name, extra=None):
    checkpoint = {
        "checkpoint": checkpoint_name,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "rss_mb": get_rss_mb(),
    }
    checkpoint.update(get_cuda_snapshot())
    if extra:
        checkpoint.update(extra)

    args.memory_checkpoints.append(checkpoint)
    args.last_successful_checkpoint = checkpoint_name
    persist_state(args)
    print(f"[CHECKPOINT] {json.dumps(checkpoint, sort_keys=True)}")


def stop_if_requested(args, stop_point):
    if args.stop_after == stop_point:
        args.status = f"stopped_after_{stop_point}"
        persist_state(args)
        print(f"[DIAGNOSTIC] Stopping after {stop_point} as requested.")
        return True
    return False


def save_completed_experiment(args, score_df, fpr, tpr, f1_score, roc_auc, tpr_at_fpr_0_01):
    fig, ax = plt.subplots(1, 1)
    ax.set_xscale("log")

    annotation = f"ROC AUC: {roc_auc:.4f}\nF1 Score: {f1_score:.2f}\nTPR at 0.01% FPR:{100 * tpr_at_fpr_0_01:.2f}%"
    display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, estimator_name=annotation)
    display.plot(ax=ax, linestyle="--")
    ax.set_title(f"{args.dataset_name} (n={len(score_df)})\nMachine Text from {args.machine_text_source}")

    fig.savefig(f"{args.experiment_path}/performance.png", bbox_inches="tight")
    plt.close(fig)

    score_df.to_csv(f"{args.experiment_path}/score_df.csv", index=False)

    args.status = "completed"
    extra = {
        "rows_scored": len(score_df),
        "end_time": datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y"),
        "f1_score": float(f1_score),
        "roc_auc": float(roc_auc),
        "tpr_at_fpr_0_01": float(tpr_at_fpr_0_01),
    }
    persist_state(args, extra=extra)


def initialize_run(args):
    args.dataset_name = args.dataset_name or args.dataset_path.rstrip("/\\").split(os.sep)[-2]
    args.machine_text_source = resolve_machine_text_source(args)
    args.threshold = resolve_threshold(args.mode)
    project_dir = os.path.dirname(os.path.abspath(__file__))
    args.job_name = (
        args.job_name
        or f"{args.dataset_name}-{args.machine_text_source}-{args.tokens_seen}-tokens".strip().replace(" ", "-")
    )
    args.experiment_path = os.path.join(project_dir, "results", args.job_name)
    os.makedirs(args.experiment_path, exist_ok=True)
    args.memory_checkpoints = []
    args.last_successful_checkpoint = None
    args.status = "running"
    persist_state(args)


def main(args):
    initialize_run(args)
    record_checkpoint(args, "process_start")

    record_checkpoint(args, "before_binoculars_init")
    bino = Binoculars(mode=args.mode, max_token_observed=args.tokens_seen)
    record_checkpoint(
        args,
        "after_binoculars_init",
        extra={
            "observer_model_device": get_model_device(bino.observer_model),
            "performer_model_device": get_model_device(bino.performer_model),
        },
    )
    if stop_if_requested(args, "model_load"):
        return

    ds = Dataset.from_json(args.dataset_path)
    record_checkpoint(args, "after_dataset_load", extra={"dataset_rows": len(ds)})
    if stop_if_requested(args, "dataset_load"):
        return

    if args.limit is not None:
        ds = ds.select(range(min(args.limit, len(ds))))
    record_checkpoint(args, "after_optional_limit", extra={"dataset_rows_after_limit": len(ds)})

    args.machine_sample_key = resolve_machine_sample_key(args, ds)
    args.machine_text_source = resolve_machine_text_source(args)
    persist_state(args)

    print("Scoring human text")
    record_checkpoint(args, "before_human_scoring_map", extra={"dataset_rows_for_human": len(ds)})
    human_scores = ds.map(
        lambda batch: {"score": bino.compute_score(batch[args.human_sample_key])},
        batched=True,
        batch_size=args.batch_size,
        remove_columns=ds.column_names,
    )
    record_checkpoint(args, "after_human_scoring_map", extra={"human_rows_scored": len(human_scores["score"])})
    if stop_if_requested(args, "human_pass"):
        return

    print("Scoring machine text")
    record_checkpoint(args, "before_machine_scoring_map", extra={"dataset_rows_for_machine": len(ds)})
    machine_scores = ds.map(
        lambda batch: {"score": bino.compute_score(batch[args.machine_sample_key])},
        batched=True,
        batch_size=args.batch_size,
        remove_columns=ds.column_names,
    )
    record_checkpoint(args, "after_machine_scoring_map", extra={"machine_rows_scored": len(machine_scores["score"])})

    score_df = convert_to_pandas(human_scores, machine_scores)
    score_df["pred"] = np.where(score_df["score"] < args.threshold, 1, 0)

    f1_score = metrics.f1_score(score_df["class"], score_df["pred"])
    score = -1 * score_df["score"]
    fpr, tpr, _ = metrics.roc_curve(y_true=score_df["class"], y_score=score, pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)
    tpr_at_fpr_0_01 = np.interp(0.01 / 100, fpr, tpr)

    record_checkpoint(args, "before_final_save", extra={"score_df_rows": len(score_df)})
    save_completed_experiment(args, score_df, fpr, tpr, f1_score, roc_auc, tpr_at_fpr_0_01)


if __name__ == "__main__":
    print("=" * 60, "START", "=" * 60)
    datasets_logging.set_verbosity_error()
    cli_args = parse_args()
    cli_args.start_time = datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")

    print("Using device:", "cuda" if torch.cuda.is_available() else "cpu")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"Resolved DEVICE_1: {detector_module.DEVICE_1}")
    print(f"Resolved DEVICE_2: {detector_module.DEVICE_2}")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"GPU Type: {torch.cuda.get_device_name(0)}")

    main(cli_args)
    print("=" * 60, "END", "=" * 60)
