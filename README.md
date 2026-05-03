# Who Wrote This? — Quantifying AI-Generated Text Across Real-World Domains

CSCI 544 (Spring 2026), Group 56 — final-report code submission. Code + artifacts for the paper [report/finalreport.pdf](report/finalreport.pdf), which scores paired pre-2022 / post-2022 corpora across five domains (Reddit, News, Amazon Reviews, arXiv, Resumes) with the Binoculars detector (Hans et al., 2024) and reports the resulting AI-prevalence shifts.

📂 **Companion data folder (Google Drive)** — canonical location for all corpora, scored artifacts, and original notebooks (not committed for size reasons): <https://drive.google.com/drive/folders/1Lc-H9QUdukV-Us_R9T4BVwXTGS3Byf66?usp=sharing>

### Submission-guideline quick map

| Guideline item | Where it's answered |
|---|---|
| How to set up the environment | [§3](#3-environment-setup) |
| Device / system used to run the code | [§2](#2-device--system-used-to-run-the-code) |
| Instructions for running the code | [§5](#5-how-to-run--reproduce-results) |
| How results are generated | [§6](#6-how-results-are-generated-methodology) (method) + [§8](#8-reported-results-must-match-the-report) (numbers) |

---

## 1. Repository layout

```
csci544-ai-detection/
├── README.md                            # this file
├── .gitignore
├── modularBinoculars/                   # the Binoculars scoring engine — runs end-to-end
│   ├── main.py                          # CLI entry point, dataset-agnostic
│   ├── src/{config,data_loader,preprocessor,scorer}.py
│   ├── requirements.txt
│   ├── job.sl, job_sample.sl            # SLURM templates (account / python parameterised)
│   ├── scripts/                         # convenience shell scripts
│   └── results/                         # small per-run analysis JSONs from report runs
├── RedditProcessor/                     # data collection: Pushshift .zst dumps → CSV → HF Hub
│   ├── parallel-script.py
│   ├── baselinescript.py
│   └── reddit_pipeline.ipynb
├── notebooks/                           # Colab notebooks (Colab + GPU)
│   ├── arxiv_research_articles_dataset.ipynb   # S3 download → LaTeX strip → chunk → Binoculars
│   ├── arxiv_fast_detect_gpt.ipynb             # Fast-DetectGPT cross-check (supporting)
│   └── binoculars_resumes.ipynb                # canonical resume scorer (4-bit NF4 + CPU spillover)
└── report/
    ├── finalreport.pdf                  # compiled report
    ├── finalreport.tex
    ├── custom.bib, refs.bib
```

---

## 2. Device / system used to run the code

| Pipeline | Hardware | OS / runtime |
|---|---|---|
| Reddit (`modularBinoculars`) at scale | NVIDIA **A40** GPU on USC CARC SLURM | Rocky Linux, Python 3.9 conda env (`binoculars`) |
| Reddit / News / Amazon Reviews / Resumes (notebook runs) | NVIDIA **T4** (Google Colab), batch size 8; resumes additionally use 4-bit NF4 quantization + `torch.float16` with CPU spillover | Ubuntu 22.04 (Colab) |
| arXiv | NVIDIA **A100** (Google Colab) | Ubuntu 22.04 (Colab) |
| Reddit local smoke tests | CPU is fine with `--dry-run`; otherwise any CUDA 11.x+ GPU | Linux, macOS, or Windows + WSL2 |

CUDA toolkit ≥ 11.x. Disk: ≥ 30 GB free (Falcon-7B + Falcon-7B-Instruct weights are ~15 GB each).

---

## 3. Environment setup

### 3a. Reddit pipeline (`modularBinoculars`) — fully reproducible

```bash
conda create -n binoculars python=3.9 -y
conda activate binoculars
cd modularBinoculars
pip install -r requirements.txt        # installs Binoculars from github.com/ahans30/Binoculars
                                       # NOT the unrelated PyPI package called `binoculars`
```

### 3b. arXiv notebooks — Colab only

Open [`notebooks/arxiv_research_articles_dataset.ipynb`](notebooks/arxiv_research_articles_dataset.ipynb) (or `arxiv_fast_detect_gpt.ipynb`) in Google Colab, attach an A100 (or T4 for Fast-DetectGPT) runtime, mount Google Drive, and set the following **Colab Secrets** (Tools → Secrets):

| Secret name | Used by | Required for |
|---|---|---|
| `AWS_ACCESS_KEY_ID` | `arxiv_research_articles_dataset.ipynb` cell 5 | downloading arXiv source LaTeX from **Amazon S3** (Simple Storage Service — Amazon's cloud file storage). The arXiv bulk archive lives in a "requester-pays" S3 bucket, meaning you (not arXiv) pay the small download bandwidth cost on your own AWS account. |
| `AWS_SECRET_ACCESS_KEY` | same | same |
| `ANTHROPIC_API_KEY` | cell 33+ | optional — only for the LLM-rewrite positive-control cells |
| `OPENAI_API_KEY` | cell 33+ | optional — same |

The notebooks read these via `google.colab.userdata.get(...)`. **Do not paste keys into cells.** The originally-distributed notebooks contained live AWS / Anthropic / OpenAI keys; those were rotated and the notebooks scrubbed before this repo was committed.

### 3c. RedditProcessor — only needed if you want to rebuild the Reddit dataset from scratch

```bash
pip install zstandard pandas huggingface_hub langdetect
# Requires the Pushshift academic-torrents .zst dump (see RedditProcessor/reddittor.torrent)
```

---

## 4. Dataset access

Pick whichever path is convenient:

| Source | Coverage | URL |
|---|---|---|
| **Hugging Face** | Reddit only (38,871 pre + 39,991 post) | <https://huggingface.co/datasets/validname/reddit-ai-detection-english-80k> |
| **Google Drive** | All five domains: Reddit, News, Amazon Reviews, arXiv, Resumes — plus the original notebooks and pre-computed scored artifacts | <https://drive.google.com/drive/folders/1Lc-H9QUdukV-Us_R9T4BVwXTGS3Byf66?usp=sharing> |
| **Pushshift torrent** | Reddit raw `.zst` archives (rebuild from scratch) | `RedditProcessor/reddittor.torrent` |

The Drive folder mirrors the layout `gdrive_data/{Reddit-Dataset/, news_data/, ReviewData/, …}`. arXiv raw and Resumes are **Drive-only**; arXiv raw can alternatively be re-downloaded from the arXiv Amazon S3 bucket via the research-articles notebook (see §3b for what S3 is).

---

## 5. How to run / reproduce results

### 5a. Reddit (the only "Run All" code path)

```bash
cd modularBinoculars
# Pipeline smoke test (no GPU needed):
python main.py --source hf --sample 100 --dry-run

# Full pre-2022 split (~38,871 rows; requires GPU):
python main.py --source hf --split pre_2022  --output reddit_pre.json

# Full post-2022 split (~39,991 rows; requires GPU):
python main.py --source hf --split post_2022 --output reddit_post.json

# SLURM — USC CARC users
sbatch --account=<your_carc_account> job.sl                          # full split
SAMPLE=500 sbatch --account=<your_carc_account> job_sample.sl        # 500-row sample
```

The accuracy threshold `0.9015310749276843` is applied automatically (`src/config.py:20`). Outputs land in `modularBinoculars/results/<name>.json`.

### 5b. News, Amazon Reviews — artifact-first lookup

If you don't need to rerun scoring, the report numbers are already in pre-computed artifacts on the Drive folder. Resumes have their own subsection (§5d).

- **News**: `news_data/news_report_outputs/news_core_summary.csv` → 3.65% pre, 2.37% post (Table 1). Per-domain: `news_domain_summary.csv`.
- **Amazon Reviews**: `ReviewData/{Raw,Processed}/*.txt` + `Computer Generated/cg_reviews.txt` (GPT-2 control).

To re-score either from scratch, see [§5e](#5e-using-modularbinoculars-as-the-shared-scoring-engine-for-any-dataset).

### 5c. arXiv — Colab + A100

1. Download the December 2021 and December 2025 arXiv source tars from Amazon S3 (the AWS-hosted arXiv bulk archive) to your mounted Drive.
2. Strip LaTeX, chunk to 250 words.
3. Score with Binoculars (Falcon-7B / Falcon-7B-Instruct, threshold `0.9015310749276843`).

```text
notebooks/arxiv_research_articles_dataset.ipynb   →  cells 1–32 sequentially
```

The Fast-DetectGPT notebook (`notebooks/arxiv_fast_detect_gpt.ipynb`) is a supporting cross-check; one comparative cell expects 2021 Fast-DetectGPT scores while the corresponding 2021 scoring call is commented out — re-enable it before running that comparison cell, or rely on the primary Binoculars notebook for the report's main numbers.

### 5d. Resumes

The Drive folder's `resumes/` subdirectory contains three files:

| File | Size | Rows | What it is |
|---|---:|---:|---|
| `Resume.csv` | 56 MB | 66,016 | Kaggle **LiveCareer** resume corpus (raw pool). Columns: `ID, Resume_str, Resume_html, Category`. The report sampled 1,037 of these. |
| `tech_resumes_dataset.jsonl` | 17 MB | 3,500 | **Tech Resumes** dataset. `ResumeID` prefixes encode the split: `REAL_*` = 2,337 human, `SYNTH_*` = 1,163 LLM-generated. |
| `binoculars_eval_data_FINAL.csv` | 1.2 MB | 4,537 | **The exact evaluation input used in the report.** Columns: `Resume_ID, Text_Block, Is_AI, Dataset`. Breakdown: `LiveCareer_DS2` = 1,037 (all human) + `Tech_Resume_DS1` = 2,337 human + 1,163 AI. Total = 4,537, matching "Final Scored: 4,537 summaries" in the report. |

**Two reproduction paths:**

**Path A (canonical — what produced the report numbers):** [`notebooks/binoculars_resumes.ipynb`](notebooks/binoculars_resumes.ipynb). One Colab cell that loads `binoculars_eval_data_FINAL.csv`, scores every row with the official Binoculars library under 4-bit NF4 quantization (`bnb_4bit_compute_dtype=torch.float16`, `bnb_4bit_use_double_quant=True`) and a 12 GiB GPU + 11 GiB CPU memory split, truncates each resume to 2000 chars, checkpoints every 100 rows, and emits `binoculars_FINAL_RESULTS.csv` with `Binoculars_Score`, `Binoculars_Prediction`, `Predicted_Is_AI`. Final cell prints overall accuracy. Runtime ≈ 1–2 hours on a T4.

**Path B (using the shared CLI):** re-score the same file through `modularBinoculars/main.py`:

```bash
# 1. Download binoculars_eval_data_FINAL.csv from the Drive folder.
# 2. Rename Text_Block -> text and copy into modularBinoculars/data/raw/:
python -c "import pandas as pd; pd.read_csv('binoculars_eval_data_FINAL.csv').rename(columns={'Text_Block':'text'}).to_csv('modularBinoculars/data/raw/resumes_eval.csv', index=False)"
# 3. Score:
cd modularBinoculars
python main.py --source local --file resumes_eval.csv --output resumes_scored.json
# 4. Aggregate flagged-rate per (Dataset, Is_AI) in the output JSON and compare to §8.
```

Path B uses identical detector parameters but does **not** apply the 2000-char truncation or the resume-specific quantization config from Path A — for tight-VRAM environments use Path A.

The "pre-2022 / post-2022" labels in the §8 results table for resumes are **not temporal** — pre = all 3,374 human resumes (LiveCareer 1,037 + REAL Tech 2,337), post = the 1,163 SYNTH (LLM-generated) Tech resumes. The 33.23% / 10.13% numbers therefore measure (false-positive rate on humans) vs. (recall on LLM resumes), and the −23.1 gap is a domain-specific Binoculars failure mode discussed in the report.

### 5e. Using `modularBinoculars` as the shared scoring engine for any dataset

`modularBinoculars/main.py` is **dataset-agnostic** — the same engine produced every Binoculars number in the report. The Reddit pipeline is just its default invocation. To re-score News, Reviews, arXiv chunks, or Resumes (or any other corpus) with identical detector parameters:

**HuggingFace dataset (any id):**

```bash
python main.py --source hf --dataset <user/dataset-id> --split <split-name> --output <name>.json
```

**Local CSV / JSONL:**

```bash
# 1. Drop the file into modularBinoculars/data/raw/
cp /path/to/news_pre_2022.csv modularBinoculars/data/raw/
# 2. Run with --source local
python main.py --source local --file news_pre_2022.csv --output news_pre.json
```

**Two schema requirements:**

- The text column must be named `text` — hardcoded as `TEXT_COLUMN` in `src/config.py:4`. Rename your column (`pandas.read_csv(...).rename(columns={"body":"text"}).to_csv(...)`) or edit the constant.
- Minimum word count is 20 (`MIN_WORDS = 20`, `src/config.py:8`). Override by editing the constant for short-text domains (e.g. some review snippets).

**Per-domain recipes (Drive sources):**

| Domain | Drive source | Schema fix | Command |
|---|---|---|---|
| News (pre) | `news_data/pre_2022/*.parquet` | concat parquets, rename body→text | `python main.py --source local --file news_pre.csv` |
| News (post) | `news_data/post_2021/*.parquet` | same | `python main.py --source local --file news_post.csv` |
| Amazon (pre) | `ReviewData/Processed/processed_2020_2021.txt` | wrap as CSV with `text` column | `python main.py --source local --file reviews_pre.csv` |
| Amazon (post) | `ReviewData/Processed/processed_2022_2023.txt` | same | `python main.py --source local --file reviews_post.csv` |
| Amazon (control) | `ReviewData/Computer Generated/processed_cg_reviews.txt` | same | `python main.py --source local --file reviews_gpt2.csv` — should reproduce ≈98.8% AI-flagged |
| arXiv (re-score) | chunked CSVs from `notebooks/arxiv_research_articles_dataset.ipynb` cells 11–17 | rename to `text` | `python main.py --source local --file arxiv_dec2021_chunks.csv` |
| Resumes | `resumes/binoculars_eval_data_FINAL.csv` (Drive) — already merged + filtered (4,537 rows) | rename `Text_Block` → `text` | `python main.py --source local --file resumes_eval.csv` (then group by `Dataset`/`Is_AI`) |

The scorer parameters (Falcon-7B / Falcon-7B-Instruct, threshold `0.9015310749276843`, batch 8) are baked in via `src/config.py`, so re-scoring is guaranteed to use identical detector settings to the report.

---

## 6. How results are generated (methodology)

Mirrors the report's methodology section so the README maps directly onto the .tex:

1. **Build paired pre-2022 / post-2022 corpora** for each domain using domain-specific sources (Pushshift Reddit dump, CC-NEWS for News, UCSD Amazon Reviews + Kaggle GPT-2 controls for Reviews, arXiv S3 for arXiv, LiveCareer + Tech + LLM-generated CSVs for Resumes).
2. **Apply consistent filtering**: 50–1500 word bounds for Reddit; 250-word chunks for arXiv; ≥15-word summaries for Resumes; HTML decode, zero-width strip, exact dedupe, English-only.
3. **Score every chunk with Binoculars** (Hans et al. 2024, [arXiv:2401.12070](https://arxiv.org/abs/2401.12070)) using the Falcon-7B observer + Falcon-7B-Instruct performer pair.
4. **Apply accuracy-mode threshold `0.9015310749276843`** (low-FPR mode `0.8536432310785527` is also available via `--mode low-fpr`).
5. **Aggregate** the AI-flagged rate (% of chunks with score ≥ threshold) per (domain, period). Report the post-minus-pre gap.
6. **Validate with controls**: GPT-2 reviews → 98.8% flagged; LLM-rewritten resumes → 100% recall; Gemma-2-2B paraphrased news → 42.47% flagged.
7. **Threshold-sensitivity sweep** (0.75 → 1.10) confirms the direction of the pre/post gap is stable across reasonable thresholds.

---

## 7. Reproducibility status matrix

| Domain | Code in repo | Data location | Runs as-is? | Notes |
|---|---|---|---|---|
| Reddit | `modularBinoculars/` | HF `validname/reddit-ai-detection-english-80k` + Drive | **Yes** | Default `main.py` invocation. SLURM scripts need `--account` override. |
| News | `modularBinoculars/` (re-score) + Drive artifacts | Drive `news_data/` | **Yes via [§5e](#5e-using-modularbinoculars-as-the-shared-scoring-engine-for-any-dataset)** | Pre-computed `news_core_summary.csv` is the fastest path; re-score by feeding parquet → CSV to `main.py`. |
| Amazon Reviews | `modularBinoculars/` (re-score) + Drive artifacts | Drive `ReviewData/` | **Yes via [§5e](#5e-using-modularbinoculars-as-the-shared-scoring-engine-for-any-dataset)** | Wrap the Processed `.txt` files into a CSV with a `text` column. |
| arXiv (raw → scored) | `notebooks/arxiv_research_articles_dataset.ipynb` | Drive + AWS S3 (requester-pays) | Colab + A100 only | End-to-end pipeline (S3 download → LaTeX strip → chunk → score) requires Colab + user-supplied keys. |
| arXiv (re-score chunks) | `modularBinoculars/` | Drive `arxiv/processed/*` | **Yes via [§5e](#5e-using-modularbinoculars-as-the-shared-scoring-engine-for-any-dataset)** | If you only need to reproduce the scoring step, feed the pre-chunked CSVs to `main.py`. |
| arXiv (FastDetectGPT) | `notebooks/arxiv_fast_detect_gpt.ipynb` | Drive | Partial | Comparative cell expects 2021 scoring that is commented out; re-enable before running. |
| Resumes | `notebooks/binoculars_resumes.ipynb` (canonical) **or** `modularBinoculars/` (re-score) | Drive `resumes/` | **Yes** | Path A (the notebook) reproduces the report numbers exactly with the resume-specific 4-bit NF4 quantization. Path B re-scores via the CLI; see §5d. |

---

## 8. Reported results (must match the report)

| Domain | Pre-2022 flagged | Post-2022 flagged | Gap | n (pre / post) |
|---|---|---|---|---|
| Reddit | 10.09% | 8.47% | −1.62 | 38,871 / 39,991 |
| News | 3.65% | 2.37% | −1.28 | 17,679 / 16,784 |
| Amazon Reviews | 16.5% | 15.23% | −1.27 | 1,000 / 1,000 |
| arXiv | 0.1% | 0.2% | +0.1 | 10,000 / 10,000 |
| Resumes † | 33.23% | 10.13% | −23.10 | 3,374 / 1,163 |

† Resumes are not split temporally. "Pre-2022" = 3,374 human resumes (1,037 LiveCareer + 2,337 REAL Tech); "Post-2022" = 1,163 SYNTH (LLM-generated) Tech resumes. The 33.23% / 10.13% are therefore (false-positive rate on humans) vs. (recall on LLM resumes); see §5d for the underlying file structure.

Plus controls: GPT-2 Amazon reviews 98.8% flagged (987/999); LLM-rewritten resumes 100% recall; Gemma-2-2B paraphrased news 42.47% flagged (7,499/17,658).

---

## 9. Known limitations / honesty disclosures

- The **arXiv pipeline** notebook is not "Run All" outside Colab + A100 — it embeds `/content/drive/...` paths and fetches from a requester-pays S3 bucket. (Re-scoring the pre-chunked CSVs via [§5e](#5e-using-modularbinoculars-as-the-shared-scoring-engine-for-any-dataset) is fully runnable, however.)
- The **Fast-DetectGPT** notebook is a supporting cross-check; one comparative cell depends on 2021 scoring that is currently commented out.
- The **News, Reviews, Resumes** domain-specific loader/cleaner glue is not packaged in this repo. The scoring engine is — `modularBinoculars/main.py` reproduces the same Falcon-7B / Falcon-7B-Instruct + threshold `0.9015310749276843` pipeline on any CSV with a `text` column ([§5e](#5e-using-modularbinoculars-as-the-shared-scoring-engine-for-any-dataset)).
- The original notebooks distributed with this work contained live API keys (AWS, Anthropic, OpenAI). Those keys have been rotated and the notebooks scrubbed; reviewers must supply their own keys via Colab Secrets.
- `gdrive_data/` is intentionally not committed (size + speed); the [Drive folder](https://drive.google.com/drive/folders/1Lc-H9QUdukV-Us_R9T4BVwXTGS3Byf66?usp=sharing) is the canonical source for raw and scored data.

---

## Citation

Hans, A., Schwarzschild, A., Cherepanova, V., Kazemi, H., Saha, A., Goldblum, M., Geiping, J., & Goldstein, T. (2024). *Spotting LLMs With Binoculars: Zero-Shot Detection of Machine-Generated Text.* arXiv:2401.12070.
