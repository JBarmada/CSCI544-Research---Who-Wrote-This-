from datasets import load_dataset
import pandas as pd
import os


def load_hf_dataset(dataset_name, split="train", sample_size=None):
    """Loads a dataset from the Hugging Face Hub.

    Args:
        dataset_name: HuggingFace dataset identifier (e.g. 'validname/reddit-ai-detection-english-80k')
        split: Dataset split to load.
        sample_size: If set, load only the first N rows via HF slice syntax (faster than
                     loading the full dataset and then truncating).
    """
    hf_split = f"{split}[:{sample_size}]" if sample_size else split
    print(f"Loading Hugging Face dataset: {dataset_name} (split='{hf_split}')...")
    dataset = load_dataset(dataset_name, split=hf_split)
    df = dataset.to_pandas()
    print(f"Loaded {len(df):,} rows.")
    return df


def load_local_dataset(file_name, sample_size=None):
    """Loads a local CSV or JSONL file from the data/raw directory.

    Args:
        file_name: Filename inside data/raw/ (e.g. 'my_data.csv').
        sample_size: If set, return only the first N rows.
    """
    from src.config import RAW_DATA_DIR
    file_path = os.path.join(RAW_DATA_DIR, file_name)

    print(f"Loading local dataset: {file_path}...")
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path, nrows=sample_size)
    elif file_path.endswith('.json') or file_path.endswith('.jsonl'):
        df = pd.read_json(file_path, lines=True)
        if sample_size:
            df = df.head(sample_size)
    else:
        raise ValueError("Unsupported format. Please use .csv or .jsonl")

    print(f"Loaded {len(df):,} rows.")
    return df