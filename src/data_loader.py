from datasets import load_dataset
import pandas as pd
import os

def load_hf_dataset(dataset_name, split="train"):
    """Loads a dataset from the Hugging Face Hub."""
    print(f"Loading Hugging Face dataset: {dataset_name}...")
    dataset = load_dataset(dataset_name, split=split)
    return dataset.to_pandas()

def load_local_dataset(file_name):
    """Loads a local CSV or JSONL file from the data/raw directory."""
    from src.config import RAW_DATA_DIR
    file_path = os.path.join(RAW_DATA_DIR, file_name)
    
    print(f"Loading local dataset: {file_path}...")
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith('.json') or file_path.endswith('.jsonl'):
        return pd.read_json(file_path, lines=True)
    else:
        raise ValueError("Unsupported format. Please use .csv or .jsonl")