# 06_make_human_eval_sheets.py
# -*- coding: utf-8 -*-

"""
Step 6. Generate blinded human evaluation sheets.

Purpose:
- Create blind A/B human evaluation sheets for group members.
- Each rater evaluates 50 items.
- A shared overlap subset is assigned to all raters for inter-rater agreement analysis.
- Baseline and emotion-aware translations are randomly assigned to Translation A/B.

Input:
    va_predictions_vadbert.csv

Output:
    human_eval_sheets/
        human_eval_rater_01.xlsx
        human_eval_rater_02.xlsx
        human_eval_rater_03.xlsx
        human_eval_master_mapping.csv

Important:
- Do NOT share human_eval_master_mapping.csv with raters.
- The master mapping file contains the hidden A/B system labels.
"""

import argparse
import random
from pathlib import Path

import pandas as pd


def assign_ab(row, rng):
    """
    Randomly assign baseline / emotion-aware translations to A/B.
    """
    if rng.random() < 0.5:
        return {
            "Translation A": row["baseline_translation"],
            "Translation B": row["emotion_aware_translation"],
            "A_system": "baseline",
            "B_system": "emotion_aware",
        }
    else:
        return {
            "Translation A": row["emotion_aware_translation"],
            "Translation B": row["baseline_translation"],
            "A_system": "emotion_aware",
            "B_system": "baseline",
        }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default="va_predictions_vadbert.csv",
        help="인간 평가표 생성을 위한 입력 CSV 파일"
    )

    parser.add_argument(
        "--output_dir",
        default="human_eval_sheets",
        help="평가표를 저장할 폴더"
    )

    parser.add_argument(
        "--num_raters",
        type=int,
        default=3,
        help="평가자 수"
    )

    parser.add_argument(
        "--items_per_rater",
        type=int,
        default=50,
        help="평가자 1인당 평가할 샘플 수"
    )

    parser.add_argument(
        "--overlap_items",
        type=int,
        default=20,
        help="모든 평가자가 공통으로 평가할 중복 샘플 수"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="무작위 배정 시드"
    )

    args = parser.parse_args()

    rng = random.Random(args.seed)

    df = pd.read_csv(args.input)

    required_cols = [
        "sample_id",
        "source_text",
        "baseline_translation",
        "emotion_aware_translation",
    ]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"필수 컬럼이 없습니다: {col}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Shuffle all samples before assignment
    df = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    if len(df) < args.overlap_items:
        raise ValueError(
            f"공통 중복 샘플 수가 전체 샘플 수보다 큽니다. "
            f"전체 샘플 수: {len(df)}, overlap_items: {args.overlap_items}"
        )

    overlap_df = df.iloc[:args.overlap_items].copy()
    remaining_df = df.iloc[args.overlap_items:].copy()

    individual_items = args.items_per_rater - args.overlap_items

    if individual_items < 0:
        raise ValueError(
            "items_per_rater는 overlap_items보다 크거나 같아야 합니다."
        )

    needed_remaining = individual_items * args.num_raters

    if len(remaining_df) < needed_remaining:
        raise ValueError(
            f"개별 배정 샘플이 부족합니다. "
            f"필요 샘플 수: {needed_remaining}, 사용 가능 샘플 수: {len(remaining_df)}"
        )

    mapping_rows = []

    print("\n========== 인간 평가표 생성 설정 ==========")
    print(f"입력 파일: {args.input}")
    print(f"출력 폴더: {args.output_dir}")
    print(f"평가자 수: {args.num_raters}")
    print(f"평가자 1인당 샘플 수: {args.items_per_rater}")
    print(f"공통 중복 샘플 수: {args.overlap_items}")
    print(f"평가자별 개별 샘플 수: {individual_items}")
    print(f"실제 커버되는 고유 샘플 수: {args.overlap_items + needed_remaining}")

    for rater_idx in range(args.num_raters):
        rater_id = f"rater_{rater_idx + 1:02d}"

        start = rater_idx * individual_items
        end = start + individual_items

        personal_df = remaining_df.iloc[start:end].copy()

        # Each rater receives:
        # overlap samples + individually assigned samples
        rater_df = pd.concat([overlap_df, personal_df], axis=0)
        rater_df = rater_df.sample(
            frac=1,
            random_state=args.seed + rater_idx
        ).reset_index(drop=True)

        sheet_rows = []

        for item_idx, row in rater_df.iterrows():
            ab = assign_ab(row, rng)
            eval_id = f"{rater_id}_item_{item_idx + 1:03d}"

            sheet_rows.append({
                "eval_id": eval_id,
                "sample_id": row["sample_id"],
                "source_text": row["source_text"],
                "Translation A": ab["Translation A"],
                "Translation B": ab["Translation B"],

                "A_semantic_fidelity_1to5": "",
                "A_emotion_consistency_1to5": "",
                "A_fluency_1to5": "",

                "B_semantic_fidelity_1to5": "",
                "B_emotion_consistency_1to5": "",
                "B_fluency_1to5": "",

                "comment": "",
            })

            mapping_rows.append({
                "rater_id": rater_id,
                "eval_id": eval_id,
                "sample_id": row["sample_id"],
                "A_system": ab["A_system"],
                "B_system": ab["B_system"],
            })

        sheet_df = pd.DataFrame(sheet_rows)

        out_path = output_dir / f"human_eval_{rater_id}.xlsx"
        sheet_df.to_excel(out_path, index=False)

        print(f"평가표 저장 완료: {out_path}")

    mapping_df = pd.DataFrame(mapping_rows)
    mapping_path = output_dir / "human_eval_master_mapping.csv"
    mapping_df.to_csv(mapping_path, index=False, encoding="utf-8-sig")

    print("\nMaster mapping 저장 완료:", mapping_path)
    print("주의: human_eval_master_mapping.csv 파일은 평가자에게 공유하지 마세요.")
    print("이 파일은 Translation A/B가 baseline인지 emotion-aware인지 복원할 때만 사용합니다.")


if __name__ == "__main__":
    main()