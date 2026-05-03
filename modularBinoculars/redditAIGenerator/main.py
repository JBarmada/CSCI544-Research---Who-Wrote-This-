import argparse
import datetime as dt
import os

from src import config
from src.checkpoint import GenerationCheckpoint
from src.generator import RedditAIGenerator
from src.io_utils import save_outputs
from src.seed_loader import load_seed_dataframe


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate Reddit AI text variants from human-written seeds."
    )
    parser.add_argument(
        "--source", choices=["hf", "local"], default="local",
        help="Seed source: Hugging Face split or local file under data/raw/."
    )
    parser.add_argument(
        "--dataset", default=config.HF_DATASET,
        help=f"Hugging Face dataset id. Default: {config.HF_DATASET}"
    )
    parser.add_argument(
        "--split", default=config.DEFAULT_SPLIT,
        help=f"Seed split to load. Default: {config.DEFAULT_SPLIT}"
    )
    parser.add_argument(
        "--file", default=config.DEFAULT_LOCAL_FILE,
        help=f"Local seed filename inside data/raw/. Default: {config.DEFAULT_LOCAL_FILE}"
    )
    parser.add_argument(
        "--sample", type=int, default=None,
        help="Optional maximum number of seed rows to load."
    )
    parser.add_argument(
        "--output-prefix", default="reddit_ai_generated",
        help="Prefix for output and checkpoint file names."
    )
    parser.add_argument(
        "--checkpoint-every", type=int, default=10,
        help="Persist checkpoint after this many completed seeds."
    )
    parser.add_argument(
        "--min-seed-words", type=int, default=config.MIN_SEED_WORDS,
        help=f"Minimum seed word count before generation. Default: {config.MIN_SEED_WORDS}"
    )
    parser.add_argument(
        "--min-generated-words", type=int, default=config.MIN_GENERATED_WORDS,
        help=f"Minimum generated word count to keep a sample. Default: {config.MIN_GENERATED_WORDS}"
    )
    parser.add_argument(
        "--batch-size", type=int, default=config.DEFAULT_BATCH_SIZE,
        help=f"Number of prompts per generation batch. Default: {config.DEFAULT_BATCH_SIZE}"
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=config.DEFAULT_MAX_NEW_TOKENS,
        help=f"Maximum new tokens per generation. Default: {config.DEFAULT_MAX_NEW_TOKENS}"
    )
    parser.add_argument(
        "--temperature", type=float, default=config.DEFAULT_TEMPERATURE,
        help=f"Sampling temperature. Default: {config.DEFAULT_TEMPERATURE}"
    )
    parser.add_argument(
        "--top-p", type=float, default=config.DEFAULT_TOP_P,
        help=f"Nucleus sampling top-p. Default: {config.DEFAULT_TOP_P}"
    )
    parser.add_argument(
        "--seed", type=int, default=config.DEFAULT_RANDOM_SEED,
        help=f"Random seed. Default: {config.DEFAULT_RANDOM_SEED}"
    )
    parser.add_argument(
        "--model", default=config.DEFAULT_MODEL_NAME,
        help=f"Generator model name. Default: {config.DEFAULT_MODEL_NAME}"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Skip model loading and emit deterministic placeholder outputs."
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Ignore any existing checkpoint and start from scratch."
    )
    return parser.parse_args()


def build_output_stem(prefix: str) -> str:
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    return f"{prefix}_{timestamp}"


def main():
    args = parse_args()
    config.ensure_directories()

    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f"{args.output_prefix}.json")
    checkpoint = GenerationCheckpoint(checkpoint_path)
    if args.no_resume:
        checkpoint.reset()
    else:
        checkpoint.load()

    df = load_seed_dataframe(
        source=args.source,
        dataset_name=args.dataset,
        split=args.split,
        file_name=args.file,
        sample_size=args.sample,
        min_seed_words=args.min_seed_words,
    )
    rows_loaded = len(df)

    generator = RedditAIGenerator(
        model_name=args.model,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        min_generated_words=args.min_generated_words,
        dry_run=args.dry_run,
        random_seed=args.seed,
    )

    generated_rows = generator.generate_rows(
        df=df,
        generated_from_split=args.split,
        checkpoint=checkpoint,
        checkpoint_every=args.checkpoint_every,
    )

    output_stem = build_output_stem(args.output_prefix)
    output_paths = save_outputs(output_stem, generated_rows)

    print("\nGeneration complete.")
    print(f"  Seed rows loaded      : {rows_loaded:,}")
    print(f"  Completed seeds       : {checkpoint.completed_count:,}")
    print(f"  Generated rows kept   : {len(generated_rows):,}")
    print(f"  CSV output            : {output_paths['csv']}")
    print(f"  JSONL output          : {output_paths['jsonl']}")
    print(f"  Checkpoint            : {checkpoint.path}")
    if args.dry_run:
        print("  Mode                  : DRY RUN")


if __name__ == "__main__":
    main()
