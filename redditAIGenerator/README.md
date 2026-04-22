# Reddit AI Generator

Generate Reddit-focused AI text datasets from the human-written `pre_2022`
seed split used elsewhere in this project.

The pipeline is designed to run full jobs on the USC SLURM GPU cluster while
still supporting tiny local dry-runs for schema validation.

## Output schema

Each generated variant is emitted as one row with these canonical columns:

- `text`
- `source_text`
- `source_id`
- `subreddit`
- `domain`
- `post_type`
- `year`
- `word_count`
- `length_bin`
- `score`
- `created_utc`
- `generation_mode`
- `model_name`
- `prompt_template`
- `prompt_version`
- `generated_from_split`
- `is_ai_generated`

`text` always refers to the generated AI text. `source_text` stores the
human-written seed used to create it.

## Generation modes

- `controlled_rewrite`
- `continuation`
- `style_conditioned`

## Local dry-run

Use dry-run mode to validate loading, fan-out, schema, and output files without
loading a model:

```bash
python main.py --source local --file sample_reddit_pre_2022.csv --sample 2 --dry-run
```

Dry-run output is written to `results/`.

## Cluster interactive validation

Run a small interactive job first:

```bash
bash scripts/run_interactive.sh --source hf --sample 8
```

## Cluster batch run

Submit a full job with `sbatch`:

```bash
sbatch job.sl
SAMPLE=128 sbatch job.sl
SOURCE=local FILE=sample_reddit_pre_2022.csv sbatch job.sl
```

## Notes

- Default seed split is `pre_2022`.
- Default generator model is `meta-llama/Llama-3.1-8B-Instruct`.
- Full runs checkpoint progress under `checkpoints/` so interrupted jobs can
  resume without regenerating completed seeds.
