import os

import pandas as pd

from src import config


def _validate_columns(df: pd.DataFrame):
    missing = [col for col in config.REQUIRED_SEED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Seed dataset is missing required columns: {missing}")


def _word_count(text: str) -> int:
    return len(str(text).split())


def _normalize_seed_dataframe(df: pd.DataFrame, min_seed_words: int) -> pd.DataFrame:
    _validate_columns(df)

    df = df.copy()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"] != ""]
    df = df[~df["text"].isin(["[deleted]", "[removed]"])]

    df["seed_word_count"] = df["text"].apply(_word_count)
    df = df[df["seed_word_count"] >= min_seed_words].copy()
    df = df.drop_duplicates(subset=["id"], keep="first")

    return df.reset_index(drop=True)


def load_seed_dataframe(
    source: str,
    dataset_name: str,
    split: str,
    file_name: str,
    sample_size: int | None,
    min_seed_words: int,
) -> pd.DataFrame:
    if source == "hf":
        from datasets import load_dataset

        hf_split = f"{split}[:{sample_size}]" if sample_size else split
        print(f"Loading Hugging Face seeds: {dataset_name} ({hf_split})")
        dataset = load_dataset(dataset_name, split=hf_split)
        df = dataset.to_pandas()
    else:
        file_path = os.path.join(config.RAW_DATA_DIR, file_name)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Local seed file not found: {file_path}")

        print(f"Loading local seeds: {file_path}")
        if file_path.endswith(".csv"):
            df = pd.read_csv(file_path, nrows=sample_size)
        elif file_path.endswith(".json") or file_path.endswith(".jsonl"):
            df = pd.read_json(file_path, lines=True)
            if sample_size:
                df = df.head(sample_size)
        else:
            raise ValueError("Unsupported local seed format. Use .csv or .jsonl")

    df = _normalize_seed_dataframe(df, min_seed_words=min_seed_words)
    print(f"Accepted seed rows after filtering: {len(df):,}")
    return df
