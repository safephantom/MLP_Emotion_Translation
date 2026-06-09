# 01_prepare_translation_inputs.py
# -*- coding: utf-8 -*-

"""
Prepare translation input data from KEMDy19-derived dataset.

Purpose:
1. Use only held-out split samples, e.g., split == test.
2. Avoid train-test leakage in translation evaluation.
3. Standardize column names for later translation and evaluation scripts.

Input:
    merged_dataset_soft_fixed.csv

Expected source columns:
    segment_id
    emotion
    valence
    arousal
    sentence_index
    sentence_korean
    endings_found
    split

Output:
    translation_inputs.csv

Output columns:
    sample_id
    segment_id
    sentence_index
    source_text
    emotion
    valence
    arousal
    ef_info
    split
"""

import argparse
import pandas as pd


def find_col(df, candidates, required=True):
    """
    Find a column from candidate names, case-insensitively.
    """
    lower_map = {c.lower(): c for c in df.columns}

    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]

    if required:
        raise ValueError(
            f"필수 컬럼을 찾을 수 없습니다.\n"
            f"후보 컬럼명: {candidates}\n"
            f"현재 컬럼명: {list(df.columns)}"
        )

    return None


def normalize_split_value(x):
    """
    Normalize split labels such as Test, test, TEST.
    """
    return str(x).strip().lower()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        help="원본 CSV 파일 경로"
    )

    parser.add_argument(
        "--output",
        default="translation_inputs.csv",
        help="출력 CSV 파일 경로"
    )

    parser.add_argument(
        "--sample_size",
        type=int,
        default=300,
        help="추출할 샘플 수. 전체 사용 시 -1 입력"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="랜덤 샘플링 시드"
    )

    parser.add_argument(
        "--split",
        default="test",
        help="사용할 split. 예: train, valid, validation, test, all"
    )

    parser.add_argument(
        "--balance_emotion",
        action="store_true",
        help="감정별 균형 샘플링을 수행할지 여부"
    )

    args = parser.parse_args()

    df = pd.read_csv(args.input)

    print("\n========== 원본 데이터 ==========")
    print("원본 크기:", df.shape)
    print("원본 컬럼:", df.columns.tolist())

    # Column mapping
    segment_id_col = find_col(df, ["segment_id"], required=True)
    sentence_index_col = find_col(df, ["sentence_index"], required=False)

    text_col = find_col(df, [
        "source_text",
        "text",
        "utterance",
        "sentence",
        "transcript",
        "sentence_korean"
    ], required=True)

    emotion_col = find_col(df, [
        "emotion",
        "label",
        "hard_label",
        "dominant_emotion",
        "emotion_label"
    ], required=True)

    valence_col = find_col(df, [
        "valence",
        "v",
        "valence_mean",
        "valence_score"
    ], required=True)

    arousal_col = find_col(df, [
        "arousal",
        "a",
        "arousal_mean",
        "arousal_score"
    ], required=True)

    ef_col = find_col(df, [
        "matched_representative_efs",
        "ef_forms_in_order",
        "ef_info",
        "ef_form",
        "ef_display_form",
        "representative_efs",
        "endings_found"
    ], required=False)

    split_col = find_col(df, ["split"], required=False)

    # Split filtering
    requested_split = args.split.strip().lower()

    if requested_split != "all":
        if split_col is None:
            raise ValueError(
                "--split 옵션을 사용했지만 원본 데이터에 split 컬럼이 없습니다."
            )

        df["_split_normalized"] = df[split_col].apply(normalize_split_value)

        print("\n========== split 분포 ==========")
        print(df["_split_normalized"].value_counts())

        before = len(df)
        df = df[df["_split_normalized"] == requested_split].copy()
        after = len(df)

        print(f"\n사용 split: {requested_split}")
        print(f"split 필터링 전: {before}")
        print(f"split 필터링 후: {after}")

        if after == 0:
            raise ValueError(
                f"split == '{requested_split}' 인 샘플이 없습니다.\n"
                f"가능한 split 값: {df['_split_normalized'].unique().tolist()}"
            )
    else:
        print("\n주의: split == all 로 설정되었습니다. 최종 평가용으로는 권장하지 않습니다.")

    # Standardized output
    out = pd.DataFrame()

    out["segment_id"] = df[segment_id_col].astype(str)

    if sentence_index_col is not None:
        out["sentence_index"] = df[sentence_index_col]
        out["sample_id"] = (
            df[segment_id_col].astype(str)
            + "_sent"
            + df[sentence_index_col].astype(str)
        )
    else:
        out["sentence_index"] = ""
        out["sample_id"] = df[segment_id_col].astype(str)

    out["source_text"] = df[text_col].astype(str)
    out["emotion"] = df[emotion_col].astype(str)
    out["valence"] = pd.to_numeric(df[valence_col], errors="coerce")
    out["arousal"] = pd.to_numeric(df[arousal_col], errors="coerce")

    if ef_col is not None:
        out["ef_info"] = df[ef_col].fillna("").astype(str)
    else:
        out["ef_info"] = ""

    if split_col is not None:
        out["split"] = df[split_col].astype(str)
    else:
        out["split"] = ""

    # Basic cleaning
    out = out.dropna(subset=["source_text", "emotion", "valence", "arousal"])
    out = out[out["source_text"].astype(str).str.strip() != ""]

    print("\n========== 정제 후 데이터 ==========")
    print("정제 후 크기:", out.shape)

    print("\n========== 감정 분포 ==========")
    print(out["emotion"].value_counts())

    # Sampling
    if args.sample_size is not None and args.sample_size > 0 and args.sample_size < len(out):

        if args.balance_emotion:
            # Emotion-balanced sampling
            emotions = sorted(out["emotion"].unique())
            n_emotions = len(emotions)
            per_emotion = args.sample_size // n_emotions
            remainder = args.sample_size % n_emotions

            sampled_parts = []

            for i, emo in enumerate(emotions):
                group = out[out["emotion"] == emo]
                n = per_emotion + (1 if i < remainder else 0)
                n = min(n, len(group))

                sampled_parts.append(
                    group.sample(n=n, random_state=args.seed)
                )

            out = pd.concat(sampled_parts, axis=0)
            out = out.sample(frac=1, random_state=args.seed).reset_index(drop=True)

            print("\n감정 균형 샘플링 수행")
            print("샘플링 후 크기:", out.shape)

        else:
            # Simple random sampling
            out = out.sample(n=args.sample_size, random_state=args.seed)
            out = out.reset_index(drop=True)

            print("\n랜덤 샘플링 수행")
            print("샘플링 후 크기:", out.shape)

    else:
        out = out.reset_index(drop=True)
        print("\n전체 샘플 사용")

    print("\n========== 최종 감정 분포 ==========")
    print(out["emotion"].value_counts())

    out.to_csv(args.output, index=False, encoding="utf-8-sig")

    print("\n저장 완료:", args.output)
    print("최종 샘플 수:", len(out))
    print("\n========== 앞 5행 ==========")
    print(out.head())
    print("\n========== 출력 컬럼 ==========")
    print(out.columns.tolist())


if __name__ == "__main__":
    main()