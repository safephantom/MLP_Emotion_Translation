# 08_human_eval_summary_simple.py
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ANNOTATORS = ["annotator_1", "annotator_2", "annotator_3"]

METRICS = {
    "semantic_fidelity": [
        "semantic_fidelity_winner_A_B_tie",
        "semantic_fidelity_A_score_1_5",
        "semantic_fidelity_B_score_1_5",
    ],
    "emotion_consistency": [
        "emotion_consistency_winner_A_B_tie",
        "emotion_consistency_A_score_1_5",
        "emotion_consistency_B_score_1_5",
    ],
    "fluency": [
        "fluency_winner_A_B_tie",
        "fluency_A_score_1_5",
        "fluency_B_score_1_5",
    ],
    "overall_preference": [
        "overall_preference_A_B_tie",
        None,
        None,
    ],
}

INVALID = {"-", "–", "—", "invalid", "제외", "불가", "평가불가"}


def read_table(path):
    path = Path(path)
    with open(path, "rb") as f:
        magic = f.read(4)

    if path.suffix.lower() in [".xlsx", ".xls"] or magic.startswith(b"PK"):
        return pd.read_excel(path, engine="openpyxl")

    for enc in ["utf-8-sig", "utf-8", "cp949", "euc-kr", "gb18030"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass

    return pd.read_csv(path, engine="python", on_bad_lines="skip")


def norm(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def norm_winner(x):
    x = norm(x)
    if x.upper() == "A":
        return "A"
    if x.upper() == "B":
        return "B"
    if x.lower() == "tie":
        return "tie"
    if x.lower() in INVALID or x in INVALID:
        return "invalid"
    return "invalid"


def to_score(x):
    x = norm(x)
    if x == "" or x.lower() in INVALID or x in INVALID:
        return np.nan
    return pd.to_numeric(x, errors="coerce")


def winner_to_condition(winner, a_cond, b_cond):
    w = norm_winner(winner)
    if w == "A":
        return a_cond
    if w == "B":
        return b_cond
    if w == "tie":
        return "tie"
    return "invalid"


def load_replacements(replacement_dir):
    replacement_dir = Path(replacement_dir)
    key_path = replacement_dir / "human_eval_replacement_answer_key.xlsx"
    if not key_path.exists():
        key_path = replacement_dir / "human_eval_replacement_answer_key.csv"

    if not key_path.exists():
        return pd.DataFrame()

    key = read_table(key_path)
    rows = []

    for ann in ANNOTATORS:
        sheet_path = replacement_dir / ann / f"{ann}_replacement_sheet.xlsx"
        if not sheet_path.exists():
            sheet_path = replacement_dir / ann / f"{ann}_replacement_sheet.csv"

        if not sheet_path.exists():
            continue

        sheet = read_table(sheet_path)
        if len(sheet) == 0:
            continue

        key_sub = key[key["annotator"] == ann].copy()
        merged = sheet.merge(
            key_sub,
            on=["eval_item_id", "sample_id"],
            how="left",
            suffixes=("", "_key"),
        )

        merged["annotator"] = ann
        merged["source_part"] = "replacement"
        rows.append(merged)

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)


def prepare_final_data(valid_existing, replacement):
    valid_existing = valid_existing.copy()
    valid_existing["source_part"] = "existing_valid"

    if len(replacement) > 0:
        df = pd.concat([valid_existing, replacement], ignore_index=True)
    else:
        df = valid_existing.copy()

    if "is_shared_item" not in df.columns and "is_shared_item_key" in df.columns:
        df["is_shared_item"] = df["is_shared_item_key"]

    df["is_shared_item"] = df["is_shared_item"].astype(str).str.lower().isin(
        ["true", "1", "yes"]
    )

    for metric, cols in METRICS.items():
        winner_col, a_score_col, b_score_col = cols

        df[f"{metric}_winner_condition"] = df.apply(
            lambda r: winner_to_condition(
                r[winner_col],
                r["A_condition"],
                r["B_condition"],
            ),
            axis=1,
        )

        if a_score_col is not None:
            df[f"{metric}_A_score"] = df[a_score_col].apply(to_score)
            df[f"{metric}_B_score"] = df[b_score_col].apply(to_score)

            df[f"{metric}_baseline_score"] = df.apply(
                lambda r: r[f"{metric}_A_score"]
                if r["A_condition"] == "baseline"
                else r[f"{metric}_B_score"],
                axis=1,
            )

            df[f"{metric}_audio_score"] = df.apply(
                lambda r: r[f"{metric}_A_score"]
                if r["A_condition"] == "audio_pred_emotion_aware"
                else r[f"{metric}_B_score"],
                axis=1,
            )

            df[f"{metric}_score_diff"] = (
                df[f"{metric}_audio_score"] - df[f"{metric}_baseline_score"]
            )

    return df


def make_winner_summary(df):
    rows = []

    for metric in METRICS:
        col = f"{metric}_winner_condition"
        counts = df[col].value_counts().to_dict()

        audio = counts.get("audio_pred_emotion_aware", 0)
        baseline = counts.get("baseline", 0)
        tie = counts.get("tie", 0)
        invalid = counts.get("invalid", 0)

        valid_n = audio + baseline + tie
        non_tie = audio + baseline

        rows.append({
            "metric": metric,
            "n_valid": valid_n,
            "audio_wins": audio,
            "baseline_wins": baseline,
            "ties": tie,
            "invalid": invalid,
            "audio_win_rate_all": audio / valid_n if valid_n else np.nan,
            "audio_win_rate_excl_ties": audio / non_tie if non_tie else np.nan,
        })

    return pd.DataFrame(rows)


def make_score_summary(df):
    rows = []

    for metric, cols in METRICS.items():
        if cols[1] is None:
            continue

        sub = df[
            [
                f"{metric}_baseline_score",
                f"{metric}_audio_score",
                f"{metric}_score_diff",
            ]
        ].dropna()

        rows.append({
            "metric": metric,
            "n": len(sub),
            "baseline_mean": sub[f"{metric}_baseline_score"].mean(),
            "audio_mean": sub[f"{metric}_audio_score"].mean(),
            "mean_diff_audio_minus_baseline": sub[f"{metric}_score_diff"].mean(),
        })

    return pd.DataFrame(rows)


def make_agreement_summary(df):
    shared = df[df["is_shared_item"] == True].copy()
    rows = []

    for metric in METRICS:
        col = f"{metric}_winner_condition"
        agree = []

        for _, sub in shared.groupby("sample_id"):
            labels = sub.drop_duplicates("annotator")[col].tolist()
            if len(labels) >= 2:
                agree.append(1 if len(set(labels)) == 1 else 0)

        rows.append({
            "metric": metric,
            "n_shared_items": len(agree),
            "complete_agreement_rate": np.mean(agree) if agree else np.nan,
        })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--valid_existing", default="human_eval_valid_existing.xlsx")
    parser.add_argument("--replacement_dir", default="human_eval_replacement_package")
    parser.add_argument("--output", default="human_eval_final_results.xlsx")
    args = parser.parse_args()

    valid_existing = read_table(args.valid_existing)
    replacement = load_replacements(args.replacement_dir)

    final_df = prepare_final_data(valid_existing, replacement)

    winner_summary = make_winner_summary(final_df)
    score_summary = make_score_summary(final_df)
    agreement_summary = make_agreement_summary(final_df)

    with pd.ExcelWriter(args.output, engine="openpyxl") as writer:
        final_df.to_excel(writer, sheet_name="merged", index=False)
        winner_summary.to_excel(writer, sheet_name="winner_summary", index=False)
        score_summary.to_excel(writer, sheet_name="score_summary", index=False)
        agreement_summary.to_excel(writer, sheet_name="agreement", index=False)

    print("\n========== 완료 ==========")
    print("output:", args.output)
    print("final annotation records:", len(final_df))

    print("\n===== winner_summary =====")
    print(winner_summary)

    print("\n===== score_summary =====")
    print(score_summary)

    print("\n===== agreement =====")
    print(agreement_summary)


if __name__ == "__main__":
    main()