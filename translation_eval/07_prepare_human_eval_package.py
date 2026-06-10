# 07_prepare_human_eval_package.py
# -*- coding: utf-8 -*-

import argparse
import random
import shutil
from pathlib import Path

import pandas as pd


def sample_from_group(df, condition_col, value, n, seed):
    sub = df[df[condition_col] == value].copy()
    if len(sub) == 0:
        return sub
    n = min(n, len(sub))
    return sub.sample(n=n, random_state=seed)


def build_candidate_pool(df, seed=42):
    """
    층화 표집:
    - audio-aware 전체 선호 승리 샘플
    - baseline 전체 선호 승리 샘플
    - tie 샘플

    최종 목표:
    - 공통 샘플 25개
    - 평가자별 개별 샘플 25개씩, 총 75개
    - 실제 고유 샘플 수 100개
    """
    condition_col = "overall_preference_winner_condition"

    audio_win = sample_from_group(df, condition_col, "audio_pred_emotion_aware", 45, seed)
    baseline_win = sample_from_group(df, condition_col, "baseline", 30, seed + 1)
    tie = sample_from_group(df, condition_col, "tie", 30, seed + 2)

    pool = pd.concat([audio_win, baseline_win, tie], ignore_index=True)
    pool = pool.drop_duplicates(subset=["sample_id"]).reset_index(drop=True)

    # 100개보다 부족하면 나머지 샘플에서 추가 표집
    if len(pool) < 100:
        rest = df[~df["sample_id"].isin(pool["sample_id"])].copy()
        need = 100 - len(pool)
        if len(rest) > 0:
            extra = rest.sample(n=min(need, len(rest)), random_state=seed + 3)
            pool = pd.concat([pool, extra], ignore_index=True)

    # 100개보다 많으면 100개로 축소
    if len(pool) > 100:
        pool = pool.sample(n=100, random_state=seed + 4).reset_index(drop=True)

    return pool


def blind_shuffle(row, rng):
    """
    A/B 순서를 무작위로 배치한다.
    평가자는 어느 번역이 baseline인지 audio-aware인지 알 수 없다.
    """
    baseline = str(row["baseline_translation"])
    audio_pred = str(row["audio_pred_emotion_aware_translation"])

    if rng.random() < 0.5:
        return {
            "translation_A": baseline,
            "translation_B": audio_pred,
            "A_condition": "baseline",
            "B_condition": "audio_pred_emotion_aware",
        }
    else:
        return {
            "translation_A": audio_pred,
            "translation_B": baseline,
            "A_condition": "audio_pred_emotion_aware",
            "B_condition": "baseline",
        }


def copy_audio(src_path, dst_dir, sample_id):
    src = Path(src_path)

    if not src.exists():
        return ""

    suffix = src.suffix
    dst_name = f"{sample_id}{suffix}"
    dst = dst_dir / dst_name

    shutil.copy2(src, dst)

    return f"audio/{dst_name}"


def make_empty_annotation_columns(row):
    row.update({
        "semantic_fidelity_winner_A_B_tie": "",
        "semantic_fidelity_A_score_1_5": "",
        "semantic_fidelity_B_score_1_5": "",

        "emotion_consistency_winner_A_B_tie": "",
        "emotion_consistency_A_score_1_5": "",
        "emotion_consistency_B_score_1_5": "",

        "fluency_winner_A_B_tie": "",
        "fluency_A_score_1_5": "",
        "fluency_B_score_1_5": "",

        "overall_preference_A_B_tie": "",
        "comment": "",
    })
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="quality_eval_audio_pred_blind.csv")
    parser.add_argument("--output_dir", default="human_eval_package")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    df = pd.read_csv(args.input)

    required = [
        "sample_id",
        "source_text",
        "segment_audio_path",
        "baseline_translation",
        "audio_pred_emotion_aware_translation",
        "overall_preference_winner_condition",
        "pred_emotion",
    ]

    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pool = build_candidate_pool(df, seed=args.seed)

    # 실제 고유 샘플 100개
    pool = pool.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    # 공통 샘플 25개: 세 평가자가 모두 평가
    shared = pool.iloc[:25].copy()

    # 개별 샘플 75개: 평가자별 25개씩 배정
    unique_pool = pool.iloc[25:100].copy()

    ann1_unique = unique_pool.iloc[0:25].copy()
    ann2_unique = unique_pool.iloc[25:50].copy()
    ann3_unique = unique_pool.iloc[50:75].copy()

    annotator_sets = {
        "annotator_1": pd.concat([shared, ann1_unique], ignore_index=True),
        "annotator_2": pd.concat([shared, ann2_unique], ignore_index=True),
        "annotator_3": pd.concat([shared, ann3_unique], ignore_index=True),
    }

    answer_key_rows = []
    manifest_rows = []

    shared_sample_ids = set(shared["sample_id"])

    for annotator, sub in annotator_sets.items():
        ann_dir = output_dir / annotator
        audio_dir = ann_dir / "audio"
        ann_dir.mkdir(parents=True, exist_ok=True)
        audio_dir.mkdir(parents=True, exist_ok=True)

        sheet_rows = []

        sub = sub.sample(frac=1, random_state=args.seed + len(annotator)).reset_index(drop=True)

        for idx, row in sub.iterrows():
            blind = blind_shuffle(row, rng)

            audio_relative_path = copy_audio(
                src_path=row["segment_audio_path"],
                dst_dir=audio_dir,
                sample_id=row["sample_id"],
            )

            eval_item_id = f"{annotator}_{idx + 1:03d}"

            sheet_row = {
                "eval_item_id": eval_item_id,
                "sample_id": row["sample_id"],
                "is_shared_item": row["sample_id"] in shared_sample_ids,
                "audio_file": audio_relative_path,
                "korean_source": row["source_text"],
                "translation_A": blind["translation_A"],
                "translation_B": blind["translation_B"],
            }

            sheet_row = make_empty_annotation_columns(sheet_row)
            sheet_rows.append(sheet_row)

            answer_key_rows.append({
                "annotator": annotator,
                "eval_item_id": eval_item_id,
                "sample_id": row["sample_id"],
                "is_shared_item": row["sample_id"] in shared_sample_ids,
                "A_condition": blind["A_condition"],
                "B_condition": blind["B_condition"],
                "pred_emotion": row.get("pred_emotion", ""),
                "overall_preference_winner_condition_from_llm": row.get("overall_preference_winner_condition", ""),
                "semantic_fidelity_winner_condition_from_llm": row.get("semantic_fidelity_winner_condition", ""),
                "emotion_consistency_winner_condition_from_llm": row.get("emotion_consistency_winner_condition", ""),
                "fluency_winner_condition_from_llm": row.get("fluency_winner_condition", ""),
                "segment_audio_path_original": row["segment_audio_path"],
            })

        sheet_df = pd.DataFrame(sheet_rows)
        sheet_path = ann_dir / f"{annotator}_sheet.csv"
        sheet_df.to_csv(sheet_path, index=False, encoding="utf-8-sig")

        manifest_rows.append({
            "annotator": annotator,
            "sheet_path": str(sheet_path),
            "audio_dir": str(audio_dir),
            "n_items": len(sheet_df),
            "n_shared": int(sheet_df["is_shared_item"].sum()),
            "n_unique": int((~sheet_df["is_shared_item"]).sum()),
        })

    answer_key_df = pd.DataFrame(answer_key_rows)
    answer_key_df.to_csv(
        output_dir / "human_eval_answer_key.csv",
        index=False,
        encoding="utf-8-sig",
    )

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(
        output_dir / "human_eval_manifest.csv",
        index=False,
        encoding="utf-8-sig",
    )

    shared[[
        "sample_id",
        "source_text",
        "pred_emotion",
        "overall_preference_winner_condition",
    ]].to_csv(
        output_dir / "shared_items_list.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n========== 완료 ==========")
    print("입력 파일:", args.input)
    print("출력 폴더:", output_dir)

    print("\n평가자별 배정 정보:")
    print(manifest_df)

    print("\n정답 키 파일:")
    print(output_dir / "human_eval_answer_key.csv")

    print("\n공통 샘플 수:", len(shared))
    print("개별 샘플 수:", len(unique_pool))
    print("실제 고유 샘플 수:", len(pool))
    print("총 평가 기록 수:", len(answer_key_df))


if __name__ == "__main__":
    main()