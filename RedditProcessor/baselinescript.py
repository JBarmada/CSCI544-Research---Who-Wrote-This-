"""
reddit_pipeline.py
==================
Collects human-written Reddit posts/comments across multiple subreddits,
stratifies them into year buckets, labels by domain, and uploads to
Hugging Face. Output files are saved locally as zipped CSVs for manual
upload to Google Drive.

Resume support
--------------
After each .zst file finishes, progress is saved to:
    reddit_output/checkpoint.json

If the script is interrupted, re-running it will:
  - Skip files that are already marked complete in the checkpoint
  - Reload the reservoir from the checkpoint so no work is lost
  - Pick up from the next unfinished file

To start completely fresh, delete reddit_output/checkpoint.json.

Year-bucket targets
-------------------
Pre-2022  →  20 000 total
  2005-2010 :  1 000
  2011-2015 :  3 000
  2016-2018 :  5 000
  2019-2021 : 11 000

Post-2022  →  20 000 total
  2022      :  3 000
  2023      :  6 500
  2024      :  6 500
  2025-2026 :  4 000

Usage
-----
1. Fill in the CONFIG section below.
2. pip install zstandard pandas huggingface_hub
3. python reddit_pipeline.py
"""

import zstandard as zstd
import json
import io
import os
import zipfile
import random
from datetime import datetime, timezone
from collections import defaultdict
from pathlib import Path

import pandas as pd

try:
    from huggingface_hub import HfApi
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("⚠  huggingface_hub not installed – HF upload disabled.")


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG  ← Edit this section
# ══════════════════════════════════════════════════════════════════════════════

SUBREDDIT_DOMAIN_MAP: dict[str, str] = {
    # Technology
    "technology":             "technology",
    "programming":            "technology",
    "MachineLearning":        "technology",
    "artificial":             "technology",
    "compsci":                "technology",
    "hardware":               "technology",
    "cybersecurity":          "technology",
    "AskTechnology":          "technology",
    # News / Politics
    "news":                   "news",
    "worldnews":              "news",
    "politics":               "news",
    "neutralnews":            "news",
    "UpliftingNews":          "news",
    "PoliticalDiscussion":    "news",
    # Science
    "science":                "science",
    "askscience":             "science",
    "EverythingScience":      "science",
    "biology":                "science",
    "physics":                "science",
    "AskScienceFiction":      "science",
    # Finance / Business
    "investing":              "finance",
    "economics":              "finance",
    "wallstreetbets":         "finance",
    "personalfinance":        "finance",
    "stocks":                 "finance",
    "investing_discussion":   "finance",
    "investingforbeginners":  "finance",
    # Entertainment
    "movies":                 "entertainment",
    "books":                  "entertainment",
    "television":             "entertainment",
    "music":                  "entertainment",
    "gaming":                 "entertainment",
    "harrypotter":            "entertainment",
    "Minecraft":              "entertainment",
    "PewdiepieSubmissions":   "entertainment",
    "moviescirclejerk":       "entertainment",
    "MovieSuggestions":       "entertainment",
    # Sports
    "CollegeBasketball":      "sports",
    # Lifestyle
    "LifeProTips":            "lifestyle",
}

# Tuples of (path_to_zst_file, subreddit_name, post_type)
# post_type must be "submission" or "comment"
INPUT_FILES: list[tuple[str, str, str]] = [
    # ── Technology ──────────────────────────────────────────────
    ("reddit/subreddits25/technology_submissions.zst",           "technology",           "submission"),
    ("reddit/subreddits25/technology_comments.zst",              "technology",           "comment"),
    ("reddit/subreddits25/Technology__submissions.zst",          "technology",           "submission"),  # note double underscore
    ("reddit/subreddits25/AskTechnology_submissions.zst",        "AskTechnology",        "submission"),
    ("reddit/subreddits25/AskTechnology_comments.zst",           "AskTechnology",        "comment"),
    # ── Science ─────────────────────────────────────────────────
    ("reddit/subreddits25/science_submissions.zst",              "science",              "submission"),
    ("reddit/subreddits25/science_comments.zst",                 "science",              "comment"),
    ("reddit/subreddits25/askscience_submissions.zst",           "askscience",           "submission"),
    ("reddit/subreddits25/askscience_comments.zst",              "askscience",           "comment"),
    ("reddit/subreddits25/AskScienceFiction_comments.zst",       "AskScienceFiction",    "comment"),
    # ── Finance ─────────────────────────────────────────────────
    ("reddit/subreddits25/investing_submissions.zst",            "investing",            "submission"),
    ("reddit/subreddits25/investing_comments.zst",               "investing",            "comment"),
    ("reddit/subreddits25/investing_discussion_submissions.zst", "investing_discussion", "submission"),
    ("reddit/subreddits25/investingforbeginners_comments.zst",   "investingforbeginners","comment"),
    # ── Entertainment ────────────────────────────────────────────
    ("reddit/subreddits25/movies_submissions.zst",               "movies",               "submission"),
    ("reddit/subreddits25/movies_comments.zst",                  "movies",               "comment"),
    ("reddit/subreddits25/moviescirclejerk_comments.zst",        "moviescirclejerk",     "comment"),
    ("reddit/subreddits25/MovieSuggestions_comments.zst",        "MovieSuggestions",     "comment"),
    ("reddit/subreddits25/gaming_comments.zst",                  "gaming",               "comment"),
    ("reddit/subreddits25/harrypotter_comments.zst",             "harrypotter",          "comment"),
    ("reddit/subreddits25/Minecraft_comments.zst",               "Minecraft",            "comment"),
    ("reddit/subreddits25/PewdiepieSubmissions_submissions.zst", "PewdiepieSubmissions", "submission"),
    # ── Sports ──────────────────────────────────────────────────
    ("reddit/subreddits25/CollegeBasketball_comments.zst",       "CollegeBasketball",    "comment"),
    # ── Lifestyle ───────────────────────────────────────────────
    ("reddit/subreddits25/LifeProTips_comments.zst",             "LifeProTips",          "comment"),
    # ── News / Politics ─────────────────────────────────────────
    ("reddit/subreddits25/PoliticalDiscussion_comments.zst",     "PoliticalDiscussion",  "comment"),
]

MIN_WORDS   = 50       # drop very short posts
MAX_WORDS   = 1_500    # drop walls of text / bot dumps
OUTPUT_DIR  = Path("reddit_output")
RANDOM_SEED = 42

# Hugging Face — set HF_TOKEN in your environment: export HF_TOKEN=hf_...
HF_REPO_ID = "validname/reddit-ai-detection"
HF_TOKEN   = os.environ.get("HF_TOKEN", "")


# ══════════════════════════════════════════════════════════════════════════════
#  YEAR BUCKETS & QUOTAS
# ══════════════════════════════════════════════════════════════════════════════

SCALE = 2.0   # 1.0 = original targets, 2.0 = double, 0.5 = half
PRE_BUCKETS: dict[tuple[int, int], int] = {
    (2005, 2010):  int(1_000  * SCALE),
    (2011, 2015):  int(3_000  * SCALE),
    (2016, 2018):  int(5_000  * SCALE),
    (2019, 2021):  int(11_000 * SCALE),
}

POST_BUCKETS: dict[tuple[int, int], int] = {
    (2022, 2022):  int(3_000  * SCALE),
    (2023, 2023):  int(6_500  * SCALE),
    (2024, 2024):  int(6_500  * SCALE),
    (2025, 2026):  int(4_000  * SCALE),
}

ALL_BUCKETS: dict[tuple[int, int], int] = {**PRE_BUCKETS, **POST_BUCKETS}

CHECKPOINT_FILE = OUTPUT_DIR / "checkpoint.json"


# ══════════════════════════════════════════════════════════════════════════════
#  UTILITIES
# ══════════════════════════════════════════════════════════════════════════════
import re
import html

def clean_text(text: str) -> str:
    # Decode HTML entities  (&amp; → &, &#x200B; → '', etc.)
    text = html.unescape(text)

    # Remove zero-width and non-printable unicode characters
    text = re.sub(r'[\u200b\u200c\u200d\ufeff]', '', text)

    # Strip URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)

    # Collapse excessive newlines (more than 2 in a row → 2)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Collapse excessive whitespace within lines
    text = re.sub(r'[ \t]{2,}', ' ', text)

    return text.strip()

def get_bucket(year: int) -> tuple[int, int] | None:
    for (start, end) in ALL_BUCKETS:
        if start <= year <= end:
            return (start, end)
    return None


def word_count(text: str) -> int:
    return len(text.split())


def length_bin(wc: int) -> str:
    if wc < 100:  return "short"
    if wc < 300:  return "medium"
    if wc < 700:  return "long"
    return "very_long"


def extract_text(obj: dict, post_type: str) -> str:
    if post_type == "submission":
        return obj.get("selftext", "").strip()
    return obj.get("body", "").strip()


def passes_filters(text: str) -> bool:
    if not text or text in {"[deleted]", "[removed]", ""}:
        return False
    wc = word_count(text)
    return MIN_WORDS <= wc <= MAX_WORDS


def read_lines_zst(file_path: str):
    with open(file_path, "rb") as f:
        dctx = zstd.ZstdDecompressor(max_window_size=2**31)
        reader = dctx.stream_reader(f)
        text_stream = io.TextIOWrapper(reader, encoding="utf-8")
        for line in text_stream:
            yield line


# ══════════════════════════════════════════════════════════════════════════════
#  CHECKPOINT
# ══════════════════════════════════════════════════════════════════════════════

def save_checkpoint(completed_files: list[str], reservoir: dict) -> None:
    """
    Write completed file list + full reservoir to JSON.
    Tuple keys like ('technology', (2019, 2021)) are stringified for JSON
    compatibility and parsed back with eval() on load.
    """
    serialisable = {str(k): v for k, v in reservoir.items()}
    payload = {
        "completed_files": completed_files,
        "reservoir":       serialisable,
    }
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    print(f"  💾 Checkpoint saved → {CHECKPOINT_FILE}")


def load_checkpoint() -> tuple[list[str], dict]:
    """
    Returns (completed_files, reservoir).
    Returns empty defaults if no checkpoint file exists.
    """
    if not CHECKPOINT_FILE.exists():
        print("  No checkpoint found — starting fresh.")
        return [], {}

    with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
        payload = json.load(f)

    completed_files = payload.get("completed_files", [])
    raw_reservoir   = payload.get("reservoir", {})

    reservoir = {}
    for k_str, records in raw_reservoir.items():
        try:
            key = eval(k_str)   # safe — we wrote these ourselves
            reservoir[key] = records
        except Exception:
            continue

    total = sum(len(v) for v in reservoir.values())
    print(f"  📂 Checkpoint loaded — {len(completed_files)} file(s) already "
          f"done, {total:,} records in reservoir.")
    return completed_files, reservoir


# ══════════════════════════════════════════════════════════════════════════════
#  RESERVOIR SAMPLER
# ══════════════════════════════════════════════════════════════════════════════

class BucketCollector:
    """
    Reservoir sampler (Vitter's Algorithm R) per (domain, year_bucket) cell.

    Every eligible post has an equal probability of appearing in the final
    set regardless of where it sits in the file. This prevents viral /
    high-scored posts — which tend to cluster near the start of dumps —
    from dominating the sample.

    Pass preloaded_reservoir to restore state from a checkpoint.
    """

    def __init__(self, quotas: dict, domains: list[str],
                 preloaded_reservoir: dict | None = None):
        self.quotas  = quotas
        self.domains = domains
        n_domains    = max(1, len(domains))
        self.seen_hashes = set() 

        self.cell_target: dict[tuple, int] = {}
        for bucket, total in quotas.items():
            per_domain = max(1, total // n_domains)
            for d in domains:
                self.cell_target[(d, bucket)] = per_domain

        if preloaded_reservoir:
            self.reservoir  = defaultdict(list, preloaded_reservoir)
            self.seen_count = defaultdict(int,
                {k: len(v) for k, v in self.reservoir.items()})
        else:
            self.reservoir  = defaultdict(list)
            self.seen_count = defaultdict(int)

    def try_add(self, record: dict) -> None:
        h = hash(record['text'])
        if h in self.seen_hashes:
            return
        self.seen_hashes.add(h)
        key    = (record["domain"], record["bucket"])
        target = self.cell_target.get(key, 0)
        if target == 0:
            return

        n = self.seen_count[key]
        self.seen_count[key] += 1

        reservoir = self.reservoir[key]
        if len(reservoir) < target:
            reservoir.append(record)
        else:
            r = random.randint(0, n)
            if r < target:
                reservoir[r] = record

    def total_collected(self) -> int:
        return sum(len(v) for v in self.reservoir.values())

    def status_table(self) -> str:
        rows = []
        for (domain, bucket), target in sorted(self.cell_target.items()):
            have = len(self.reservoir[(domain, bucket)])
            bar  = "█" * int(20 * have / max(target, 1))
            rows.append(
                f"  {domain:<15s} {str(bucket):<20s} "
                f"{have:>5d}/{target:<5d}  {bar}"
            )
        return "\n".join(rows)

    def to_dataframe(self) -> pd.DataFrame:
        all_records = []
        for records in self.reservoir.values():
            all_records.extend(records)
        df = pd.DataFrame(all_records)
        if "bucket" in df.columns:
            df = df.drop(columns=["bucket"])
        return df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
#  HUGGING FACE UPLOAD
# ══════════════════════════════════════════════════════════════════════════════

def upload_to_hf(pre_df, post_df):
    if not HF_AVAILABLE or not HF_TOKEN or 'your-hf-username' in HF_REPO_ID:
        print('  HF upload skipped (token or repo not configured).')
        return

    from datasets import Dataset, DatasetDict

    dataset = DatasetDict({
        'pre_2022':  Dataset.from_pandas(pre_df,  preserve_index=False),
        'post_2022': Dataset.from_pandas(post_df, preserve_index=False),
    })

    dataset.push_to_hub(
        HF_REPO_ID,
        token=HF_TOKEN,
        private=False,
    )
    print(f'  ✓ Uploaded to HF with splits: hf://datasets/{HF_REPO_ID}')

def upload_readme_to_hf(pre_df: pd.DataFrame, post_df: pd.DataFrame) -> None:
    if not HF_AVAILABLE or not HF_TOKEN or "your-hf-username" in HF_REPO_ID:
        return

    total  = len(pre_df) + len(post_df)
    readme = f"""\
---
language: en
license: other
task_categories:
  - text-classification
tags:
  - ai-detection
  - reddit
  - human-written
  - nlp
size_categories:
  - 10K<n<100K
---

# Reddit AI-Detection Dataset

Human-written Reddit posts and comments collected for AI-generated text
detection research (CSCI 544 – *Who Wrote This?*).

All records in the **pre-2022** split pre-date widespread LLM deployment
and can be treated as ground-truth human-authored text for detector
calibration.

## Splits

| File | Records | Period |
|------|--------:|--------|
| `data/reddit_pre_2022.zip` | {len(pre_df):,} | 2005 – 2021 |
| `data/reddit_post_2022.zip` | {len(post_df):,} | 2022 – 2026 |
| `data/reddit_combined.zip` | {total:,} | 2005 – 2026 |

## Schema

| Column | Type | Description |
|--------|------|-------------|
| `text` | str | Post / comment body |
| `subreddit` | str | Source subreddit |
| `domain` | str | technology · news · science · finance · entertainment |
| `post_type` | str | `submission` or `comment` |
| `year` | int | UTC year of posting |
| `word_count` | int | Approximate word count |
| `length_bin` | str | short / medium / long / very_long |
| `score` | int | Reddit score at collection time |
| `created_utc` | int | Unix timestamp |
| `id` | str | Reddit post ID |

## Citation

Data originally collected by Pushshift / u/raiderbdev, packaged by u/Watchful1.
"""
    readme_path = OUTPUT_DIR / "README.md"
    readme_path.write_text(readme, encoding="utf-8")
    upload_to_hf(readme_path, "README.md")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline() -> None:
    random.seed(RANDOM_SEED)
    OUTPUT_DIR.mkdir(exist_ok=True)

    # ── Resume: load checkpoint if it exists ─────────────────────────────────
    print("═" * 65)
    print(" STARTUP — Checking for existing checkpoint")
    print("═" * 65)
    completed_files, preloaded_reservoir = load_checkpoint()

    domains   = sorted(set(SUBREDDIT_DOMAIN_MAP.values()))
    collector = BucketCollector(ALL_BUCKETS, domains,
                                preloaded_reservoir or None)

    # ── Phase 1: Read & collect ───────────────────────────────────────────────
    print()
    print("═" * 65)
    print(" PHASE 1 — Reading .zst files and collecting samples")
    print("═" * 65)

    for file_path, subreddit, post_type in INPUT_FILES:

        if file_path in completed_files:
            print(f"\n✓  Already done (checkpoint): {file_path}")
            continue

        if not Path(file_path).exists():
            print(f"\n⚠  File not found, skipping: {file_path}")
            continue

        domain = SUBREDDIT_DOMAIN_MAP.get(subreddit, "other")
        print(f"\n▶  {file_path}")
        print(f"   subreddit={subreddit}  domain={domain}  type={post_type}")

        lines_read = 0
        for line in read_lines_zst(file_path):
            lines_read += 1
            try:
                obj  = json.loads(line)
                text = extract_text(obj, post_type)
                text = clean_text(text)

                if not passes_filters(text):
                    continue

                ts      = obj.get("created_utc", 0)
                created = datetime.fromtimestamp(ts, tz=timezone.utc)
                year    = created.year
                bucket  = get_bucket(year)
                if bucket is None:
                    continue

                wc = word_count(text)
                record = {
                    "text":        text,
                    "subreddit":   subreddit,
                    "domain":      domain,
                    "post_type":   post_type,
                    "year":        year,
                    "bucket":      bucket,
                    "word_count":  wc,
                    "length_bin":  length_bin(wc),
                    "score":       obj.get("score", 0),
                    "created_utc": int(ts),
                    "id":          obj.get("id", ""),
                }
                collector.try_add(record)

            except Exception:
                continue

            if lines_read % 250_000 == 0:
                print(f"   Lines: {lines_read:>10,} | "
                      f"Collected: {collector.total_collected():>7,}")

        print(f"   Done.  Lines read: {lines_read:,} | "
              f"Collected so far: {collector.total_collected():,}")

        # Mark file complete and persist progress
        completed_files.append(file_path)
        save_checkpoint(completed_files, dict(collector.reservoir))

    print("\n── Collection status ──────────────────────────────────────")
    print(collector.status_table())
    print()

    # ── Phase 2: Build DataFrames & save locally ──────────────────────────────
    print("═" * 65)
    print(" PHASE 2 — Building and saving datasets")
    print("═" * 65)

    df      = collector.to_dataframe()
    pre_df  = df[df["year"] <  2022].copy().reset_index(drop=True)
    post_df = df[df["year"] >= 2022].copy().reset_index(drop=True)

    print(f"  Pre-2022  : {len(pre_df):,} records")
    print(f"  Post-2022 : {len(post_df):,} records")
    print(f"  Total     : {len(df):,} records")

    zip_files: list[Path] = []

    for split_name, split_df in [
        ("pre_2022",  pre_df),
        ("post_2022", post_df),
        ("combined",  df),
    ]:
        csv_path = OUTPUT_DIR / f"reddit_{split_name}.csv"
        zip_path = OUTPUT_DIR / f"reddit_{split_name}.zip"

        split_df.to_csv(csv_path, index=False)

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED,
                             compresslevel=9) as zf:
            zf.write(csv_path, arcname=csv_path.name)

        size_kb = zip_path.stat().st_size // 1024
        print(f"  Saved {zip_path}  ({size_kb:,} KB)")
        zip_files.append(zip_path)

    print(f"\n  📁 Output folder: {OUTPUT_DIR.resolve()}")
    print("     Upload the .zip files to Google Drive manually.")

    # ── Phase 3: Hugging Face ─────────────────────────────────────────────────
    print()
    print("═" * 65)
    print(" PHASE 3 — Hugging Face upload")
    print("═" * 65)

    for zip_path in zip_files:
        print(f"\n  Uploading {zip_path.name} …")
        upload_to_hf(zip_path, f"data/{zip_path.name}")

    upload_readme_to_hf(pre_df, post_df)

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print("═" * 65)
    print(" SUMMARY")
    print("═" * 65)
    print(f"\n  Total records  : {len(df):,}")
    print(f"  Pre-2022       : {len(pre_df):,}")
    print(f"  Post-2022      : {len(post_df):,}\n")

    print("  By domain:")
    for domain, grp in df.groupby("domain"):
        print(f"    {domain:<18s} {len(grp):>7,}")

    print("\n  By year:")
    year_counts = df.groupby("year").size()
    for year, count in year_counts.items():
        bar = "▪" * min(40, count // 100)
        print(f"    {year}  {count:>6,}  {bar}")

    print("\n  By length bin:")
    for bin_name, grp in df.groupby("length_bin"):
        print(f"    {bin_name:<12s} {len(grp):>7,}")

    print("\n✓ Pipeline complete.\n")


# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    run_pipeline()