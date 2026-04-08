import os
from src import config
from src.data_loader import load_hf_dataset, load_local_dataset
from src.preprocessor import preprocess_data
from src.scorer import score_dataframe

def main():
    # Ensure output directory exists
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    # --- STEP 1: LOAD DATA ---
    # To use Hugging Face, uncomment the line below and add your dataset name:
    # df = load_hf_dataset("username/reddit_dataset_name")
    
    # Or to use a local file, ensure it is placed in data/raw/:
    df = load_local_dataset("my_reddit_data.csv")

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
        mode=config.BINO_MODE,
        threshold=config.BINO_THRESHOLD
    )

    # --- STEP 4: SAVE ---
    output_path = os.path.join(config.RESULTS_DIR, "reddit_scored_results.csv")
    df_scored.to_csv(output_path, index=False)
    print(f"\nPipeline complete! All metadata columns and new scores saved to: {output_path}")

if __name__ == "__main__":
    main()