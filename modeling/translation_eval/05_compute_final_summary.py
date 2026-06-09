# 05_compute_final_summary.py
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import pandas as pd
from scipy import stats


def clean_pair(df, baseline_col, target_col):
    baseline = pd.to_numeric(df[baseline_col], errors="coerce")
    target = pd.to_numeric(df[target_col], errors="coerce")

    valid = baseline.notna() & target.notna()

    return baseline[valid].to_numpy(), target[valid].to_numpy()


def paired_t_test(baseline, target):
    if len(baseline) < 3:
        return None

    result = stats.ttest_rel(target, baseline, nan_policy="omit")
    return float(result.pvalue)


def wilcoxon_test(baseline, target):
    if len(baseline) < 3:
        return None

    try:
        result = stats.wilcoxon(target, baseline)
        return float(result.pvalue)
    except ValueError:
        return None


def bootstrap_ci(baseline, target, n_boot=5000, seed=42):
    rng = np.random.default_rng(seed)
    diff = target - baseline

    if len(diff) < 3:
        return None, None

    boot_means = []

    for _ in range(n_boot):
        sample = rng.choice(diff, size=len(diff), replace=True)
        boot_means.append(sample.mean())

    low, high = np.percentile(boot_means, [2.5, 97.5])
    return float(low), float(high)


def cohen_dz(baseline, target):
    diff = target - baseline

    if len(diff) < 2:
        return None

    sd = diff.std(ddof=1)

    if sd == 0:
        return None

    return float(diff.mean() / sd)


def summarize_metric(df, metric_name, baseline_col, target_col):
    baseline, target = clean_pair(df, baseline_col, target_col)

    if len(baseline) < 3:
        return None

    diff = target - baseline
    ci_low, ci_high = bootstrap_ci(baseline, target)

    return {
        "metric": metric_name,
        "n": len(baseline),
        "baseline_mean": baseline.mean(),
        "emotion_aware_mean": target.mean(),
        "difference_emotion_aware_minus_baseline": diff.mean(),
        "baseline_std": baseline.std(ddof=1),
        "emotion_aware_std": target.std(ddof=1),
        "diff_95ci_low": ci_low,
        "diff_95ci_high": ci_high,
        "paired_t_p": paired_t_test(baseline, target),
        "wilcoxon_p": wilcoxon_test(baseline, target),
        "cohen_dz": cohen_dz(baseline, target),
    }


def build_summary(df):
    metric_pairs = {
        "R_VA": (
            "baseline_R_VA",
            "emotion_aware_R_VA"
        ),
        "semantic_fidelity": (
            "baseline_semantic_fidelity",
            "emotion_aware_semantic_fidelity"
        ),
        "emotion_consistency": (
            "baseline_emotion_consistency",
            "emotion_aware_emotion_consistency"
        ),
        "fluency": (
            "baseline_fluency",
            "emotion_aware_fluency"
        ),
    }

    rows = []

    for metric_name, (baseline_col, target_col) in metric_pairs.items():
        if baseline_col not in df.columns or target_col not in df.columns:
            print(f"[Skip] Missing columns for {metric_name}")
            continue

        row = summarize_metric(df, metric_name, baseline_col, target_col)

        if row is not None:
            rows.append(row)

    return pd.DataFrame(rows)


def add_interpretation(summary_df):
    interpretations = []

    for _, row in summary_df.iterrows():
        diff = row["difference_emotion_aware_minus_baseline"]
        p = row["wilcoxon_p"]

        if pd.isna(p):
            sig = "not_tested"
        elif p < 0.001:
            sig = "***"
        elif p < 0.01:
            sig = "**"
        elif p < 0.05:
            sig = "*"
        else:
            sig = "n.s."

        if diff > 0 and sig != "n.s." and sig != "not_tested":
            interp = "emotion-aware significantly higher"
        elif diff > 0:
            interp = "emotion-aware higher trend"
        elif diff < 0 and sig != "n.s." and sig != "not_tested":
            interp = "baseline significantly higher"
        elif diff < 0:
            interp = "baseline higher trend"
        else:
            interp = "no difference"

        interpretations.append(interp)

    summary_df["significance"] = summary_df["wilcoxon_p"].apply(
        lambda p: (
            "***" if pd.notna(p) and p < 0.001 else
            "**" if pd.notna(p) and p < 0.01 else
            "*" if pd.notna(p) and p < 0.05 else
            "n.s." if pd.notna(p) else
            "not_tested"
        )
    )

    summary_df["interpretation"] = interpretations

    return summary_df


def round_numeric(df):
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].round(6)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="quality_eval_deepseek_blind.csv")
    parser.add_argument("--summary_output", default="final_eval_summary.csv")
    parser.add_argument("--emotion_output", default="final_eval_by_emotion.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    summary_df = build_summary(df)
    summary_df = add_interpretation(summary_df)
    summary_df = round_numeric(summary_df)
    summary_df.to_csv(args.summary_output, index=False, encoding="utf-8-sig")

    print("\n========== Overall summary ==========")
    print(summary_df)
    print("\n저장 완료:", args.summary_output)

    if "emotion" in df.columns:
        emotion_rows = []

        for emotion, group in df.groupby("emotion"):
            group_summary = build_summary(group)

            if len(group_summary) == 0:
                continue

            group_summary.insert(0, "emotion", emotion)
            emotion_rows.append(group_summary)

        if emotion_rows:
            emotion_df = pd.concat(emotion_rows, axis=0).reset_index(drop=True)
            emotion_df = add_interpretation(emotion_df)
            emotion_df = round_numeric(emotion_df)
            emotion_df.to_csv(args.emotion_output, index=False, encoding="utf-8-sig")

            print("\n========== Emotion-wise summary ==========")
            print(emotion_df)
            print("\n감정별 요약 저장 완료:", args.emotion_output)


if __name__ == "__main__":
    main()