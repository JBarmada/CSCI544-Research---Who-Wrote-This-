import argparse
import os
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
        "--sample", type=int, default=None,
        help="Number of rows to use. Omit to run on the full dataset."
    )
    parser.add_argument(
        "--mode", choices=["accuracy", "low-fpr"], default="accuracy",
        help="Detection mode: 'accuracy' (F1-optimised) or 'low-fpr' (0.01%% FPR). (default: accuracy)"
    )
    parser.add_argument(
        "--output", default="results.csv",
        help="Output filename inside results/. (default: results.csv)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve threshold from mode
    threshold = (
        config.BINO_THRESHOLD if args.mode == "accuracy"
        else config.BINO_THRESHOLD_LOW_FPR
    )

    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    # --- STEP 1: LOAD DATA ---
    if args.source == "hf":
        df = load_hf_dataset(args.dataset, sample_size=args.sample)
    else:
        if not args.file:
            raise ValueError("--file is required when --source=local")
        df = load_local_dataset(args.file, sample_size=args.sample)

    # --- STEP 2: PREPROCESS ---
    df_clean = preprocess_data(
        df,
        text_column=config.TEXT_COLUMN,
        min_words=config.MIN_WORDS
    )

    # --- STEP 3: SCORE ---
    df_scored = score_dataframe(
        df_clean,
        text_column=config.TEXT_COLUMN,
        batch_size=config.BATCH_SIZE,
        mode=args.mode,
        threshold=threshold
    )

    # --- STEP 4: SAVE ---
    output_path = os.path.join(config.RESULTS_DIR, args.output)
    df_scored.to_csv(output_path, index=False)
    print(f"\nPipeline complete! Results saved to: {output_path}")
    print(f"  Rows scored : {len(df_scored):,}")
    print(f"  Mode        : {args.mode}  (threshold={threshold})")


if __name__ == "__main__":
    main()
