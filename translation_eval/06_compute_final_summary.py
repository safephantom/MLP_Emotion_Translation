# 06_compute_final_summary.py
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import pandas as pd
from scipy import stats


def bootstrap_ci_mean_diff(diff, n_boot=10000, ci=95, seed=42):
    rng = np.random.default_rng(seed)
    diff = np.asarray(diff, dtype=float)
    diff = diff[~np.isnan(diff)]

    if len(diff) == 0:
        return np.nan, np.nan

    boot_means = []
    for _ in range(n_boot):
        sample = rng.choice(diff, size=len(diff), replace=True)
        boot_means.append(np.mean(sample))

    alpha = (100 - ci) / 2
    lower = np.percentile(boot_means, alpha)
    upper = np.percentile(boot_means, 100 - alpha)
    return lower, upper


def cohens_dz(diff):
    diff = np.asarray(diff, dtype=float)
    diff = diff[~np.isnan(diff)]

    if len(diff) < 2:
        return np.nan

    sd = np.std(diff, ddof=1)
    if sd == 0:
        return 0.0

    return np.mean(diff) / sd


def paired_tests(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = ~(np.isnan(x) | np.isnan(y))
    x = x[mask]
    y = y[mask]

    diff = y - x

    if len(diff) < 2:
        return {
            "n": len(diff),
            "mean_baseline": np.nan,
            "mean_audio_pred": np.nan,
            "mean_diff": np.nan,
            "std_diff": np.nan,
            "paired_t": np.nan,
            "paired_t_p": np.nan,
            "wilcoxon_stat": np.nan,
            "wilcoxon_p": np.nan,
            "cohens_dz": np.nan,
            "bootstrap_ci_low": np.nan,
            "bootstrap_ci_high": np.nan,
        }

    t_stat, t_p = stats.ttest_rel(y, x)

    try:
        w_stat, w_p = stats.wilcoxon(diff)
    except Exception:
        w_stat, w_p = np.nan, np.nan

    ci_low, ci_high = bootstrap_ci_mean_diff(diff)

    return {
        "n": len(diff),
        "mean_baseline": float(np.mean(x)),
        "mean_audio_pred": float(np.mean(y)),
        "mean_diff": float(np.mean(diff)),
        "std_diff": float(np.std(diff, ddof=1)),
        "paired_t": float(t_stat),
        "paired_t_p": float(t_p),
        "wilcoxon_stat": float(w_stat) if not pd.isna(w_stat) else np.nan,
        "wilcoxon_p": float(w_p) if not pd.isna(w_p) else np.nan,
        "cohens_dz": float(cohens_dz(diff)),
        "bootstrap_ci_low": float(ci_low),
        "bootstrap_ci_high": float(ci_high),
    }


def count_winners(df, col):
    counts = df[col].value_counts(dropna=False).to_dict()

    audio = counts.get("audio_pred_emotion_aware", 0)
    baseline = counts.get("baseline", 0)
    tie = counts.get("tie", 0)
    total = len(df)

    non_tie = audio + baseline
    if non_tie > 0:
        audio_win_rate_excluding_ties = audio / non_tie
        baseline_win_rate_excluding_ties = baseline / non_tie
    else:
        audio_win_rate_excluding_ties = np.nan
        baseline_win_rate_excluding_ties = np.nan

    return {
        "metric": col.replace("_winner_condition", ""),
        "n": total,
        "audio_pred_wins": audio,
        "baseline_wins": baseline,
        "ties": tie,
        "audio_pred_win_rate_all": audio / total if total else np.nan,
        "baseline_win_rate_all": baseline / total if total else np.nan,
        "tie_rate_all": tie / total if total else np.nan,
        "audio_pred_win_rate_excluding_ties": audio_win_rate_excluding_ties,
        "baseline_win_rate_excluding_ties": baseline_win_rate_excluding_ties,
    }


def summarize_by_emotion(df):
    rows = []

    for emotion, sub in df.groupby("pred_emotion"):
        row = {
            "pred_emotion": emotion,
            "n": len(sub),
        }

        if "baseline_R_VA" in sub.columns and "audio_pred_R_VA" in sub.columns:
            test = paired_tests(sub["baseline_R_VA"], sub["audio_pred_R_VA"])
            row.update({
                "R_VA_baseline_mean": test["mean_baseline"],
                "R_VA_audio_pred_mean": test["mean_audio_pred"],
                "R_VA_diff_mean": test["mean_diff"],
                "R_VA_paired_t_p": test["paired_t_p"],
                "R_VA_wilcoxon_p": test["wilcoxon_p"],
                "R_VA_cohens_dz": test["cohens_dz"],
                "R_VA_ci_low": test["bootstrap_ci_low"],
                "R_VA_ci_high": test["bootstrap_ci_high"],
            })

        for col in [
            "semantic_fidelity_winner_condition",
            "emotion_consistency_winner_condition",
            "fluency_winner_condition",
            "overall_preference_winner_condition",
        ]:
            if col in sub.columns:
                c = count_winners(sub, col)
                prefix = c["metric"]

                row[f"{prefix}_audio_pred_wins"] = c["audio_pred_wins"]
                row[f"{prefix}_baseline_wins"] = c["baseline_wins"]
                row[f"{prefix}_ties"] = c["ties"]
                row[f"{prefix}_audio_win_rate_excl_ties"] = c["audio_pred_win_rate_excluding_ties"]

        rows.append(row)

    return pd.DataFrame(rows).sort_values("n", ascending=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="quality_eval_audio_pred_blind.csv")
    parser.add_argument("--summary_output", default="final_eval_audio_pred_summary.csv")
    parser.add_argument("--by_emotion_output", default="final_eval_audio_pred_by_emotion.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    summary_rows = []

    # 1. R_VA summary
    if "baseline_R_VA" in df.columns and "audio_pred_R_VA" in df.columns:
        test = paired_tests(df["baseline_R_VA"], df["audio_pred_R_VA"])
        test["section"] = "automatic_metric"
        test["metric"] = "zscore_aligned_R_VA"
        summary_rows.append(test)

    if "baseline_R_VA_raw" in df.columns and "audio_pred_R_VA_raw" in df.columns:
        test = paired_tests(df["baseline_R_VA_raw"], df["audio_pred_R_VA_raw"])
        test["section"] = "automatic_metric"
        test["metric"] = "raw_R_VA"
        summary_rows.append(test)

    # 2. Blind judge winner counts
    judge_cols = [
        "semantic_fidelity_winner_condition",
        "emotion_consistency_winner_condition",
        "fluency_winner_condition",
        "overall_preference_winner_condition",
    ]

    for col in judge_cols:
        if col in df.columns:
            row = count_winners(df, col)
            row["section"] = "blind_pairwise_judge"
            summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    # Put section/metric first
    front_cols = [c for c in ["section", "metric"] if c in summary_df.columns]
    other_cols = [c for c in summary_df.columns if c not in front_cols]
    summary_df = summary_df[front_cols + other_cols]

    summary_df.to_csv(args.summary_output, index=False, encoding="utf-8-sig")

    # 3. By-emotion summary
    if "pred_emotion" in df.columns:
        by_emotion_df = summarize_by_emotion(df)
        by_emotion_df.to_csv(args.by_emotion_output, index=False, encoding="utf-8-sig")
    else:
        by_emotion_df = pd.DataFrame()

    print("\n========== Done ==========")
    print("input:", args.input)
    print("summary_output:", args.summary_output)
    print("by_emotion_output:", args.by_emotion_output)
    print("rows:", len(df))

    print("\n===== Final Summary =====")
    print(summary_df)

    if not by_emotion_df.empty:
        print("\n===== By Emotion Summary =====")
        print(by_emotion_df.head(20))


if __name__ == "__main__":
    main()