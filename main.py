import argparse
import json
import os
import datetime
from src import config
from src.data_loader import load_hf_dataset, load_local_dataset
from src.preprocessor import preprocess_data
from src.scorer import score_dataframe


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the Binoculars LLM-text detector on a dataset."
    )
    parser.add_argument(
        "--source", choices=["hf", "local"], default="hf",
        help="Data source: 'hf' for HuggingFace Hub, 'local' for a file in data/raw/. (default: hf)"
    )
    parser.add_argument(
        "--dataset", default=config.HF_DATASET,
        help=f"HuggingFace dataset name. (default: {config.HF_DATASET})"
    )
    parser.add_argument(
        "--file", default=None,
        help="Local filename inside data/raw/ (e.g. my_data.csv). Required when --source=local."
    )
    parser.add_argument(
        "--split", default="post_2022",
        help="HuggingFace dataset split to use. For this dataset: 'post_2022' or 'pre_2022'. (default: post_2022)"
    )
    parser.add_argument(
        "--sample", type=int, default=None,
        help="Number of rows to use. Omit to run on the full dataset."
    )
    parser.add_argument(
        "--mode", choices=["accuracy", "low-fpr"], default="accuracy",
        help="Detection mode: 'accuracy' (F1-optimised) or 'low-fpr' (0.01%% FPR). (default: accuracy)"
    )
    parser.add_argument(
        "--output", default="results.json",
        help="Output filename inside results/. (default: results.json)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Skip model inference. Loads and preprocesses data only — use to verify the pipeline works before committing to a GPU allocation."
    )
    return parser.parse_args()


def save_results_json(df_scored, args, threshold, rows_loaded, rows_after_preprocess, output_path):
    """Save scored results as a structured JSON file with run metadata and per-row records."""

    ai_count = (df_scored["binoculars_prediction"] == "Most likely AI-generated").sum()
    human_count = (df_scored["binoculars_prediction"] == "Most likely human-generated").sum()
    error_count = (df_scored["binoculars_prediction"] == "Error").sum()
    scored = df_scored["binoculars_score"].notna().sum()
    valid_scores = df_scored["binoculars_score"].dropna()

    source_info = (
        {"type": "huggingface", "dataset": args.dataset}
        if args.source == "hf"
        else {"type": "local", "file": args.file}
    )

    output = {
        "run": {
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "source": source_info,
            "sample_size_requested": args.sample,
            "mode": args.mode,
            "threshold": threshold,
            "observer_model": config.OBSERVER_MODEL,
            "performer_model": config.PERFORMER_MODEL,
            "batch_size": config.BATCH_SIZE,
            "min_words_filter": config.MIN_WORDS,
        },
        "summary": {
            "rows_loaded": rows_loaded,
            "rows_after_preprocessing": rows_after_preprocess,
            "rows_scored_successfully": int(scored),
            "rows_with_errors": int(error_count),
            "ai_generated_count": int(ai_count),
            "human_generated_count": int(human_count),
            "ai_generated_pct": round(ai_count / scored * 100, 2) if scored > 0 else None,
            "score_mean": round(float(valid_scores.mean()), 6) if len(valid_scores) > 0 else None,
            "score_std": round(float(valid_scores.std()), 6) if len(valid_scores) > 0 else None,
            "score_min": round(float(valid_scores.min()), 6) if len(valid_scores) > 0 else None,
            "score_max": round(float(valid_scores.max()), 6) if len(valid_scores) > 0 else None,
        },
        "results": df_scored.to_dict(orient="records"),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)


def main():
    args = parse_args()

    threshold = (
        config.BINO_THRESHOLD if args.mode == "accuracy"
        else config.BINO_THRESHOLD_LOW_FPR
    )

    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    # --- STEP 1: LOAD DATA ---
    if args.source == "hf":
        df = load_hf_dataset(args.dataset, split=args.split, sample_size=args.sample)
    else:
        if not args.file:
            raise ValueError("--file is required when --source=local")
        df = load_local_dataset(args.file, sample_size=args.sample)

    rows_loaded = len(df)

    # --- STEP 2: PREPROCESS ---
    df_clean = preprocess_data(
        df,
        text_column=config.TEXT_COLUMN,
        min_words=config.MIN_WORDS
    )

    rows_after_preprocess = len(df_clean)

    # --- STEP 3: SCORE (skipped in dry-run mode) ---
    if args.dry_run:
        print("\n[DRY RUN] Skipping model inference.")
        df_scored = df_clean.copy()
        df_scored["binoculars_score"] = None
        df_scored["binoculars_prediction"] = "DRY_RUN"
    else:
        df_scored = score_dataframe(
            df_clean,
            text_column=config.TEXT_COLUMN,
            batch_size=config.BATCH_SIZE,
            mode=args.mode,
            threshold=threshold
        )

    # --- STEP 4: SAVE ---
    output_name = args.output if args.output.endswith(".json") else args.output.replace(".csv", "") + ".json"
    output_path = os.path.join(config.RESULTS_DIR, output_name)

    save_results_json(df_scored, args, threshold, rows_loaded, rows_after_preprocess, output_path)

    ai_count = (df_scored["binoculars_prediction"] == "Most likely AI-generated").sum()
    print(f"\nPipeline complete! Results saved to: {output_path}")
    print(f"  Rows loaded     : {rows_loaded:,}")
    print(f"  Rows scored     : {rows_after_preprocess:,}")
    if not args.dry_run:
        print(f"  AI-generated    : {ai_count:,}  ({ai_count/rows_after_preprocess*100:.1f}%)")
    print(f"  Mode            : {args.mode}  (threshold={threshold})")
    if args.dry_run:
        print("  [DRY RUN] No inference was performed. Re-run without --dry-run to score.")


if __name__ == "__main__":
    main()
