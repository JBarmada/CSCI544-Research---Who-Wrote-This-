"""
analyze.py — Aggregate and summarize Binoculars results JSON.

Usage:
    python analyze.py                                      # reads results/sample_results.json
    python analyze.py --input results/full_results.json
    python analyze.py --input results/sample_results.json --output results/analysis.json
"""

import argparse
import json
import os
import pandas as pd


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate Binoculars detection results.")
    parser.add_argument("--input",  default="results/sample_results.json",
                        help="Path to the results JSON file produced by main.py.")
    parser.add_argument("--output", default=None,
                        help="Where to save the analysis JSON. Defaults to <input_stem>_analysis.json.")
    return parser.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────────

def group_stats(df, group_col):
    """Return per-group counts, AI%, and score stats as a list of dicts."""
    if group_col not in df.columns:
        return None

    rows = []
    for name, grp in df.groupby(group_col, dropna=False):
        valid = grp["binoculars_score"].dropna()
        ai    = (grp["binoculars_prediction"] == "Most likely AI-generated").sum()
        total = len(grp)
        rows.append({
            group_col:          str(name),
            "total":            int(total),
            "ai_count":         int(ai),
            "human_count":      int(total - ai),
            "ai_pct":           round(ai / total * 100, 2) if total else None,
            "score_mean":       round(float(valid.mean()), 6) if len(valid) else None,
            "score_std":        round(float(valid.std()),  6) if len(valid) > 1 else None,
            "score_min":        round(float(valid.min()),  6) if len(valid) else None,
            "score_max":        round(float(valid.max()),  6) if len(valid) else None,
        })
    return sorted(rows, key=lambda r: r["total"], reverse=True)


def score_histogram(series, bins=20):
    """Return a simple histogram as a list of {bin_start, bin_end, count} dicts."""
    valid = series.dropna()
    if len(valid) == 0:
        return []
    hist_series = pd.cut(valid, bins=bins, include_lowest=True).value_counts(sort=False)
    result = []
    for interval, count in hist_series.items():
        result.append({
            "bin_start": round(interval.left,  4),
            "bin_end":   round(interval.right, 4),
            "count":     int(count),
        })
    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Load results
    print(f"Loading: {args.input}")
    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    run_meta = data.get("run", {})
    records  = data.get("results", [])

    if not records:
        print("No results found in file.")
        return

    df = pd.DataFrame(records)

    # Ensure score is numeric
    df["binoculars_score"] = pd.to_numeric(df["binoculars_score"], errors="coerce")

    total   = len(df)
    scored  = df["binoculars_score"].notna().sum()
    ai      = (df["binoculars_prediction"] == "Most likely AI-generated").sum()
    human   = (df["binoculars_prediction"] == "Most likely human-generated").sum()
    errors  = (df["binoculars_prediction"] == "Error").sum()
    valid_scores = df["binoculars_score"].dropna()

    print(f"  Rows          : {total:,}")
    print(f"  AI-generated  : {ai:,}  ({ai/scored*100:.1f}%)" if scored else "")
    print(f"  Human         : {human:,}  ({human/scored*100:.1f}%)" if scored else "")
    print(f"  Score mean    : {valid_scores.mean():.4f}" if len(valid_scores) else "")

    # ── Build analysis output ──────────────────────────────────────────────
    analysis = {
        "source_file": args.input,
        "run": run_meta,

        "overall": {
            "total_rows":      int(total),
            "rows_scored":     int(scored),
            "rows_errors":     int(errors),
            "ai_count":        int(ai),
            "human_count":     int(human),
            "ai_pct":          round(ai / scored * 100, 2) if scored else None,
            "score_mean":      round(float(valid_scores.mean()), 6) if len(valid_scores) else None,
            "score_std":       round(float(valid_scores.std()),  6) if len(valid_scores) > 1 else None,
            "score_median":    round(float(valid_scores.median()), 6) if len(valid_scores) else None,
            "score_min":       round(float(valid_scores.min()),  6) if len(valid_scores) else None,
            "score_max":       round(float(valid_scores.max()),  6) if len(valid_scores) else None,
            "score_25pct":     round(float(valid_scores.quantile(0.25)), 6) if len(valid_scores) else None,
            "score_75pct":     round(float(valid_scores.quantile(0.75)), 6) if len(valid_scores) else None,
        },

        "by_subreddit":  group_stats(df, "subreddit"),
        "by_post_type":  group_stats(df, "post_type"),
        "by_year":       group_stats(df, "year"),
        "by_length_bin": group_stats(df, "length_bin"),
        "by_domain":     group_stats(df, "domain"),

        "score_histogram": score_histogram(df["binoculars_score"]),
    }

    # ── Save ──────────────────────────────────────────────────────────────
    if args.output:
        out_path = args.output
    else:
        stem     = os.path.splitext(args.input)[0]
        out_path = stem + "_analysis.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False, default=str)

    print(f"\nAnalysis saved to: {out_path}")


if __name__ == "__main__":
    main()
