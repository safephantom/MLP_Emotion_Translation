# 07_5_prepare_replacement_for_invalid_items.py
# -*- coding: utf-8 -*-

import argparse
import random
import shutil
from pathlib import Path

import pandas as pd


ANNOTATORS = ["annotator_1", "annotator_2", "annotator_3"]

WINNER_COLS = [
    "semantic_fidelity_winner_A_B_tie",
    "emotion_consistency_winner_A_B_tie",
    "fluency_winner_A_B_tie",
    "overall_preference_A_B_tie",
]

SCORE_COLS = [
    "semantic_fidelity_A_score_1_5",
    "semantic_fidelity_B_score_1_5",
    "emotion_consistency_A_score_1_5",
    "emotion_consistency_B_score_1_5",
    "fluency_A_score_1_5",
    "fluency_B_score_1_5",
]

INVALID_VALUES = {"-", "–", "—", "invalid", "제외", "불가", "평가불가"}


def read_table(path):
    path = Path(path)

    with open(path, "rb") as f:
        magic = f.read(4)

    # xlsx 파일인데 확장자가 csv로 잘못 저장된 경우까지 처리
    if path.suffix.lower() in [".xlsx", ".xls"] or magic.startswith(b"PK"):
        return pd.read_excel(path, engine="openpyxl")

    for enc in ["utf-8-sig", "utf-8", "cp949", "euc-kr", "gb18030", "latin1"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue

    return pd.read_csv(path, encoding="utf-8-sig", engine="python", on_bad_lines="skip")


def write_xlsx(df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(path, index=False)


def find_sheet(package_dir, annotator):
    base = Path(package_dir) / annotator

    for name in [
        f"{annotator}_sheet.xlsx",
        f"{annotator}_sheet.csv",
    ]:
        p = base / name
        if p.exists():
            return p

    raise FileNotFoundError(f"평가자 시트를 찾을 수 없습니다: {annotator}")


def find_answer_key(package_dir):
    base = Path(package_dir)

    for name in [
        "human_eval_answer_key.xlsx",
        "human_eval_answer_key.csv",
    ]:
        p = base / name
        if p.exists():
            return p

    raise FileNotFoundError("human_eval_answer_key 파일을 찾을 수 없습니다.")


def norm(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def is_invalid(row):
    for col in WINNER_COLS + SCORE_COLS:
        if col in row.index and norm(row[col]).lower() in INVALID_VALUES:
            return True
    return False


def count_hangul(text):
    text = norm(text)
    return sum(1 for ch in text if "가" <= ch <= "힣")


def map_winner_to_condition(winner, a_condition, b_condition):
    w = norm(winner)

    if w.upper() == "A":
        return a_condition
    if w.upper() == "B":
        return b_condition
    if w.lower() == "tie":
        return "tie"
    if w in INVALID_VALUES:
        return "invalid"

    return ""


def merge_sheet_with_key(sheet, key, annotator):
    key_sub = key[key["annotator"] == annotator].copy()

    merged = sheet.merge(
        key_sub,
        on=["eval_item_id", "sample_id"],
        how="left",
        suffixes=("", "_key"),
    )

    merged["annotator"] = annotator

    if "is_shared_item" not in merged.columns and "is_shared_item_key" in merged.columns:
        merged["is_shared_item"] = merged["is_shared_item_key"]

    merged["is_shared_item"] = merged["is_shared_item"].astype(str).str.lower().isin(
        ["true", "1", "yes"]
    )

    merged["is_invalid"] = merged.apply(is_invalid, axis=1)

    for col in WINNER_COLS:
        if col in merged.columns:
            metric = col.replace("_winner_A_B_tie", "")
            merged[f"{metric}_winner_condition_human"] = merged.apply(
                lambda r: map_winner_to_condition(
                    r[col],
                    r.get("A_condition", ""),
                    r.get("B_condition", ""),
                ),
                axis=1,
            )

    return merged


def resolve_audio_path(path_str, project_root):
    p = Path(str(path_str))

    if p.exists():
        return p

    p2 = project_root / p
    if p2.exists():
        return p2

    p3 = Path.cwd() / p
    if p3.exists():
        return p3

    return p


def copy_audio(src_path, dst_dir, sample_id, project_root):
    src = resolve_audio_path(src_path, project_root)

    if not src.exists():
        return ""

    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    dst = dst_dir / f"{sample_id}{src.suffix}"
    shutil.copy2(src, dst)

    return f"audio/{dst.name}"


def blind_shuffle(row, rng):
    baseline = str(row["baseline_translation"])
    audio_pred = str(row["audio_pred_emotion_aware_translation"])

    if rng.random() < 0.5:
        return baseline, audio_pred, "baseline", "audio_pred_emotion_aware"

    return audio_pred, baseline, "audio_pred_emotion_aware", "baseline"


def empty_annotation_cols(row):
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


def filter_candidates(candidates, used_ids, min_korean_chars, min_duration_sec):
    pool = candidates.copy()
    pool["sample_id"] = pool["sample_id"].astype(str)

    pool = pool[~pool["sample_id"].isin(used_ids)].copy()
    pool["korean_char_count"] = pool["source_text"].apply(count_hangul)
    pool = pool[pool["korean_char_count"] >= min_korean_chars].copy()

    if "duration_sec" in pool.columns:
        dur = pd.to_numeric(pool["duration_sec"], errors="coerce")
        pool = pool[dur >= min_duration_sec].copy()

    pool = pool.dropna(
        subset=[
            "segment_audio_path",
            "baseline_translation",
            "audio_pred_emotion_aware_translation",
        ]
    )

    return pool.reset_index(drop=True)


def select_replacement(candidates, used_ids, preferred_condition, seed, min_korean_chars, min_duration_sec):
    pool = filter_candidates(
        candidates=candidates,
        used_ids=used_ids,
        min_korean_chars=min_korean_chars,
        min_duration_sec=min_duration_sec,
    )

    if preferred_condition and "overall_preference_winner_condition" in pool.columns:
        preferred = pool[pool["overall_preference_winner_condition"] == preferred_condition]
        if len(preferred) > 0:
            pool = preferred

    if len(pool) == 0:
        raise RuntimeError("교체 가능한 후보 샘플이 부족합니다.")

    return pool.sample(n=1, random_state=seed).iloc[0]


def make_replacement_item(annotator, idx, row, is_shared, rng, audio_dir, project_root):
    trans_a, trans_b, cond_a, cond_b = blind_shuffle(row, rng)

    eval_item_id = f"{annotator}_R{idx:03d}"

    audio_file = copy_audio(
        src_path=row["segment_audio_path"],
        dst_dir=audio_dir,
        sample_id=row["sample_id"],
        project_root=project_root,
    )

    sheet_row = {
        "eval_item_id": eval_item_id,
        "sample_id": row["sample_id"],
        "is_shared_item": is_shared,
        "audio_file": audio_file,
        "korean_source": row["source_text"],
        "translation_A": trans_a,
        "translation_B": trans_b,
    }

    sheet_row = empty_annotation_cols(sheet_row)

    key_row = {
        "annotator": annotator,
        "eval_item_id": eval_item_id,
        "sample_id": row["sample_id"],
        "is_shared_item": is_shared,
        "A_condition": cond_a,
        "B_condition": cond_b,
        "pred_emotion": row.get("pred_emotion", ""),
        "overall_preference_winner_condition_from_llm": row.get(
            "overall_preference_winner_condition", ""
        ),
        "semantic_fidelity_winner_condition_from_llm": row.get(
            "semantic_fidelity_winner_condition", ""
        ),
        "emotion_consistency_winner_condition_from_llm": row.get(
            "emotion_consistency_winner_condition", ""
        ),
        "fluency_winner_condition_from_llm": row.get(
            "fluency_winner_condition", ""
        ),
        "segment_audio_path_original": row["segment_audio_path"],
        "is_replacement": 1,
    }

    return sheet_row, key_row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package_dir", default="human_eval_package")
    parser.add_argument("--candidate_file", default="quality_eval_audio_pred_blind.csv")
    parser.add_argument("--output_dir", default="human_eval_replacement_package")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_korean_chars", type=int, default=6)
    parser.add_argument("--min_duration_sec", type=float, default=1.0)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    cwd = Path.cwd()
    project_root = cwd.parent if cwd.name == "translation_eval" else cwd

    package_dir = Path(args.package_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    answer_key = read_table(find_answer_key(package_dir))
    candidates = read_table(args.candidate_file)

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
        if col not in candidates.columns:
            raise ValueError(f"candidate_file에 필요한 열이 없습니다: {col}")

    merged_all = []

    for annotator in ANNOTATORS:
        sheet = read_table(find_sheet(package_dir, annotator))
        merged = merge_sheet_with_key(sheet, answer_key, annotator)
        merged_all.append(merged)

    merged_raw = pd.concat(merged_all, ignore_index=True)

    # 공통 샘플은 한 명이라도 invalid이면 세 명 모두 제외
    invalid_shared_ids = set(
        merged_raw[
            (merged_raw["is_shared_item"] == True)
            & (merged_raw["is_invalid"] == True)
        ]["sample_id"].astype(str)
    )

    merged_raw["invalid_group"] = merged_raw.apply(
        lambda r: str(r["sample_id"]) in invalid_shared_ids or bool(r["is_invalid"]),
        axis=1,
    )

    valid_existing = merged_raw[merged_raw["invalid_group"] == False].copy()
    invalid_items = merged_raw[merged_raw["invalid_group"] == True].copy()

    write_xlsx(merged_raw, "human_eval_merged_raw.xlsx")
    write_xlsx(valid_existing, "human_eval_valid_existing.xlsx")
    write_xlsx(invalid_items, "human_eval_invalid_items.xlsx")

    used_ids = set(merged_raw["sample_id"].astype(str))

    replacement_key_rows = []
    manifest_rows = []

    # shared invalid replacement: 동일한 기존 shared sample은 동일한 새 sample로 대체
    shared_replacement_map = {}

    invalid_shared_originals = (
        invalid_items[invalid_items["is_shared_item"] == True]
        [["sample_id", "overall_preference_winner_condition_from_llm"]]
        .drop_duplicates(subset=["sample_id"])
    )

    for i, old_row in enumerate(invalid_shared_originals.itertuples(index=False), start=1):
        repl = select_replacement(
            candidates=candidates,
            used_ids=used_ids,
            preferred_condition=getattr(
                old_row,
                "overall_preference_winner_condition_from_llm",
                None,
            ),
            seed=args.seed + i,
            min_korean_chars=args.min_korean_chars,
            min_duration_sec=args.min_duration_sec,
        )

        used_ids.add(str(repl["sample_id"]))
        shared_replacement_map[str(old_row.sample_id)] = repl

    for annotator in ANNOTATORS:
        ann_invalid = invalid_items[invalid_items["annotator"] == annotator].copy()

        ann_dir = output_dir / annotator
        audio_dir = ann_dir / "audio"
        ann_dir.mkdir(parents=True, exist_ok=True)
        audio_dir.mkdir(parents=True, exist_ok=True)

        sheet_rows = []

        for idx, (_, invalid_row) in enumerate(ann_invalid.iterrows(), start=1):
            old_sample_id = str(invalid_row["sample_id"])

            if bool(invalid_row["is_shared_item"]):
                repl = shared_replacement_map[old_sample_id]
                is_shared = True
            else:
                repl = select_replacement(
                    candidates=candidates,
                    used_ids=used_ids,
                    preferred_condition=invalid_row.get(
                        "overall_preference_winner_condition_from_llm",
                        None,
                    ),
                    seed=args.seed + 1000 + idx,
                    min_korean_chars=args.min_korean_chars,
                    min_duration_sec=args.min_duration_sec,
                )

                used_ids.add(str(repl["sample_id"]))
                is_shared = False

            sheet_row, key_row = make_replacement_item(
                annotator=annotator,
                idx=idx,
                row=repl,
                is_shared=is_shared,
                rng=rng,
                audio_dir=audio_dir,
                project_root=project_root,
            )

            key_row["replaces_old_eval_item_id"] = invalid_row["eval_item_id"]
            key_row["replaces_old_sample_id"] = invalid_row["sample_id"]

            sheet_rows.append(sheet_row)
            replacement_key_rows.append(key_row)

        sheet_df = pd.DataFrame(sheet_rows)

        sheet_path = ann_dir / f"{annotator}_replacement_sheet.xlsx"
        write_xlsx(sheet_df, sheet_path)

        manifest_rows.append({
            "annotator": annotator,
            "replacement_sheet_path": str(sheet_path),
            "n_replacement_items": len(sheet_df),
            "n_shared_replacements": int(sheet_df["is_shared_item"].sum()) if len(sheet_df) else 0,
            "n_unique_replacements": int((~sheet_df["is_shared_item"]).sum()) if len(sheet_df) else 0,
        })

    replacement_key = pd.DataFrame(replacement_key_rows)
    manifest = pd.DataFrame(manifest_rows)

    write_xlsx(replacement_key, output_dir / "human_eval_replacement_answer_key.xlsx")
    write_xlsx(manifest, output_dir / "human_eval_replacement_manifest.xlsx")

    print("\n========== 완료 ==========")
    print("전체 기존 평가 기록:", len(merged_raw))
    print("유효 기존 평가 기록:", len(valid_existing))
    print("제외 평가 기록:", len(invalid_items))
    print("invalid shared sample 수:", len(invalid_shared_ids))

    print("\n평가자별 보충 항목 수:")
    print(manifest)

    print("\n출력 폴더:", output_dir)
    print("평가자에게는 각자의 replacement_sheet.xlsx와 audio 폴더만 전달하세요.")
    print("answer_key 파일은 연구자만 보관하세요.")


if __name__ == "__main__":
    main()