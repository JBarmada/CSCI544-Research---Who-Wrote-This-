import json
import os

import pandas as pd

from src import config


OUTPUT_COLUMNS = [
    "text",
    "source_text",
    "source_id",
    "subreddit",
    "domain",
    "post_type",
    "year",
    "word_count",
    "length_bin",
    "score",
    "created_utc",
    "generation_mode",
    "model_name",
    "prompt_template",
    "prompt_version",
    "generated_from_split",
    "is_ai_generated",
]


def save_outputs(output_stem: str, rows: list[dict]) -> dict[str, str]:
    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=OUTPUT_COLUMNS)
    else:
        df = df[OUTPUT_COLUMNS]

    csv_path = os.path.join(config.RESULTS_DIR, f"{output_stem}.csv")
    jsonl_path = os.path.join(config.RESULTS_DIR, f"{output_stem}.jsonl")

    df.to_csv(csv_path, index=False)
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for row in df.to_dict(orient="records"):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return {"csv": csv_path, "jsonl": jsonl_path}
