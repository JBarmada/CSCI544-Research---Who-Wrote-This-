import argparse
import json
import os

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze a completed Binoculars reproduction run.",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to a run directory under results/ or directly to score_df.csv.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output path. Defaults to <run_dir>/analysis.json.",
    )
    return parser.parse_args()


def resolve_paths(user_input):
    if os.path.isdir(user_input):
        run_dir = user_input
        csv_path = os.path.join(run_dir, "score_df.csv")
    else:
        csv_path = user_input
        run_dir = os.path.dirname(os.path.abspath(csv_path))
    metadata_path = os.path.join(run_dir, "experiments_details.json")
    return run_dir, csv_path, metadata_path


def class_stats(df):
    return {
        "rows": int(len(df)),
        "score_mean": float(df["score"].mean()),
        "score_std": float(df["score"].std()),
        "score_min": float(df["score"].min()),
        "score_max": float(df["score"].max()),
        "predicted_machine": int((df["pred"] == 1).sum()),
        "predicted_human": int((df["pred"] == 0).sum()),
    }


def main():
    args = parse_args()
    run_dir, csv_path, metadata_path = resolve_paths(args.input)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find score_df.csv at: {csv_path}")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Could not find experiments_details.json at: {metadata_path}")

    df = pd.read_csv(csv_path)
    with open(metadata_path, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    human_df = df[df["class"] == 0]
    machine_df = df[df["class"] == 1]

    analysis = {
        "run_dir": run_dir,
        "metadata": metadata,
        "overall": {
            "rows_total": int(len(df)),
            "rows_human": int(len(human_df)),
            "rows_machine": int(len(machine_df)),
            "predicted_machine": int((df["pred"] == 1).sum()),
            "predicted_human": int((df["pred"] == 0).sum()),
            "score_mean": float(df["score"].mean()),
            "score_std": float(df["score"].std()),
            "score_min": float(df["score"].min()),
            "score_max": float(df["score"].max()),
        },
        "human_class": class_stats(human_df),
        "machine_class": class_stats(machine_df),
        "confusion": {
            "true_human_pred_human": int(((df["class"] == 0) & (df["pred"] == 0)).sum()),
            "true_human_pred_machine": int(((df["class"] == 0) & (df["pred"] == 1)).sum()),
            "true_machine_pred_human": int(((df["class"] == 1) & (df["pred"] == 0)).sum()),
            "true_machine_pred_machine": int(((df["class"] == 1) & (df["pred"] == 1)).sum()),
        },
    }

    output_path = args.output or os.path.join(run_dir, "analysis.json")
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(analysis, handle, indent=2)

    print(f"Analysis saved to: {output_path}")


if __name__ == "__main__":
    main()
