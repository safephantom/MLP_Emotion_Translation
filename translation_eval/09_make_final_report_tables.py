# 09_make_final_report_tables.py
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def read_table(path, sheet_name=0):
    if path is None:
        return pd.DataFrame()

    path = Path(path)
    if not path.exists():
        return pd.DataFrame()

    if path.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(path, sheet_name=sheet_name, engine="openpyxl")

    for enc in ["utf-8-sig", "utf-8", "cp949", "euc-kr", "gb18030"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass

    return pd.read_csv(path, engine="python", on_bad_lines="skip")


def first_existing(paths):
    for p in paths:
        if Path(p).exists():
            return p
    return None


def summarize_llm_judge(df):
    if df.empty:
        return pd.DataFrame()

    rows = []

    for col in [
        "semantic_fidelity_winner_condition",
        "emotion_consistency_winner_condition",
        "fluency_winner_condition",
        "overall_preference_winner_condition",
    ]:
        if col not in df.columns:
            continue

        counts = df[col].value_counts().to_dict()

        audio = counts.get("audio_pred_emotion_aware", 0)
        base = counts.get("baseline", 0)
        tie = counts.get("tie", 0)
        non_tie = audio + base

        rows.append({
            "metric": col.replace("_winner_condition", ""),
            "audio_wins": audio,
            "baseline_wins": base,
            "ties": tie,
            "audio_win_rate_excluding_ties": audio / non_tie if non_tie else np.nan,
        })

    return pd.DataFrame(rows)


def get_condition_translation(row, condition):
    if row.get("A_condition", "") == condition:
        return row.get("translation_A", "")
    if row.get("B_condition", "") == condition:
        return row.get("translation_B", "")
    return ""


def get_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def make_examples(df, n=5):
    if df.empty:
        return pd.DataFrame()

    df = df.copy()

    if "korean_source" not in df.columns and "source_text" in df.columns:
        df["korean_source"] = df["source_text"]

    df["baseline_translation"] = df.apply(
        lambda r: get_condition_translation(r, "baseline"),
        axis=1,
    )

    df["audio_aware_translation"] = df.apply(
        lambda r: get_condition_translation(r, "audio_pred_emotion_aware"),
        axis=1,
    )

    emotion_col = get_col(df, [
        "emotion_consistency_winner_condition_human",
        "emotion_consistency_winner_condition",
    ])

    semantic_col = get_col(df, [
        "semantic_fidelity_winner_condition_human",
        "semantic_fidelity_winner_condition",
    ])

    overall_col = get_col(df, [
        "overall_preference_winner_condition_human",
        "overall_preference_winner_condition",
    ])

    parts = []

    if emotion_col:
        sub = df[df[emotion_col] == "audio_pred_emotion_aware"].copy()
        if len(sub):
            sub = sub.sample(n=min(n, len(sub)), random_state=42)
            sub["case_type"] = "audio_aware_wins_emotion_consistency"
            parts.append(sub)

    if semantic_col:
        sub = df[df[semantic_col] == "baseline"].copy()
        if len(sub):
            sub = sub.sample(n=min(3, len(sub)), random_state=43)
            sub["case_type"] = "baseline_wins_semantic_fidelity"
            parts.append(sub)

    if overall_col:
        sub = df[df[overall_col] == "tie"].copy()
        if len(sub):
            sub = sub.sample(n=min(3, len(sub)), random_state=44)
            sub["case_type"] = "overall_tie"
            parts.append(sub)

    if not parts:
        return pd.DataFrame()

    out = pd.concat(parts, ignore_index=True)

    keep = [
        "case_type",
        "sample_id",
        "annotator",
        "pred_emotion",
        "korean_source",
        "baseline_translation",
        "audio_aware_translation",
        semantic_col,
        emotion_col,
        overall_col,
        "comment",
    ]

    keep = [c for c in keep if c and c in out.columns]
    return out[keep]


def write_markdown(path):
    text = """# Final Results Write-up

## Main conclusion

The audio-model emotion-aware translation mainly improves emotion consistency rather than general translation quality.

The automatic V/A preservation metric showed only a weak positive tendency. However, both LLM-based blind evaluation and human evaluation indicated that the audio-aware condition was more effective in preserving emotional nuance and interactional stance.

## Recommended interpretation

The proposed method should not be described as uniformly improving translation quality. Instead, it should be interpreted as selectively improving emotion consistency. Semantic fidelity showed no clear improvement and may slightly favor the baseline, while fluency remained largely unchanged.

## Korean summary

audio-model emotion-aware 번역은 일반적인 번역 품질 전반을 향상시킨다기보다는, 감정 일관성과 화자의 상호작용적 태도 보존에 선택적으로 기여하는 것으로 해석된다.
"""
    Path(path).write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_xlsx", default="final_report_tables.xlsx")
    parser.add_argument("--output_md", default="final_results_writeup.md")
    args = parser.parse_args()

    auto_summary_path = first_existing([
        "final_eval_audio_pred_summary.xlsx",
        "final_eval_audio_pred_summary.csv",
    ])

    auto_by_emotion_path = first_existing([
        "final_eval_audio_pred_by_emotion.xlsx",
        "final_eval_audio_pred_by_emotion.csv",
    ])

    llm_raw_path = first_existing([
        "quality_eval_audio_pred_blind.xlsx",
        "quality_eval_audio_pred_blind.csv",
    ])

    human_full_summary_path = first_existing([
        "human_eval_final_summary.xlsx",
    ])

    human_simple_path = first_existing([
        "human_eval_final_results.xlsx",
    ])

    human_merged_path = first_existing([
        "human_eval_final_merged.xlsx",
    ])

    agreement_path = first_existing([
        "human_eval_inter_annotator_agreement.xlsx",
    ])

    auto_summary = read_table(auto_summary_path)
    auto_by_emotion = read_table(auto_by_emotion_path)

    llm_raw = read_table(llm_raw_path)
    llm_summary = summarize_llm_judge(llm_raw)

    if human_full_summary_path:
        human_summary = read_table(human_full_summary_path)
    elif human_simple_path:
        winner = read_table(human_simple_path, sheet_name="winner_summary")
        score = read_table(human_simple_path, sheet_name="score_summary")
        winner["section"] = "human_pairwise_winner"
        score["section"] = "human_score"
        human_summary = pd.concat([winner, score], ignore_index=True)
    else:
        human_summary = pd.DataFrame()

    if human_merged_path:
        human_merged = read_table(human_merged_path)
    elif human_simple_path:
        human_merged = read_table(human_simple_path, sheet_name="merged")
    else:
        human_merged = pd.DataFrame()

    if agreement_path:
        agreement = read_table(agreement_path)
    elif human_simple_path:
        agreement = read_table(human_simple_path, sheet_name="agreement")
    else:
        agreement = pd.DataFrame()

    examples = make_examples(human_merged)

    with pd.ExcelWriter(args.output_xlsx, engine="openpyxl") as writer:
        auto_summary.to_excel(writer, sheet_name="automatic_RVA", index=False)
        auto_by_emotion.to_excel(writer, sheet_name="automatic_by_emotion", index=False)
        llm_summary.to_excel(writer, sheet_name="llm_blind_summary", index=False)
        human_summary.to_excel(writer, sheet_name="human_summary", index=False)
        agreement.to_excel(writer, sheet_name="human_agreement", index=False)
        examples.to_excel(writer, sheet_name="qualitative_examples", index=False)

    write_markdown(args.output_md)

    print("\n========== 완료 ==========")
    print("output xlsx:", args.output_xlsx)
    print("output md:", args.output_md)
    print("\nSheets:")
    print("- automatic_RVA")
    print("- automatic_by_emotion")
    print("- llm_blind_summary")
    print("- human_summary")
    print("- human_agreement")
    print("- qualitative_examples")
    print("\nqualitative examples:", len(examples))


if __name__ == "__main__":
    main()