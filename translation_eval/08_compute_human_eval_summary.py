# 08_compute_human_eval_summary.py
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


ANNOTATORS = ["annotator_1", "annotator_2", "annotator_3"]

METRICS = {
    "semantic_fidelity": {
        "winner": "semantic_fidelity_winner_A_B_tie",
        "A": "semantic_fidelity_A_score_1_5",
        "B": "semantic_fidelity_B_score_1_5",
        "llm": "semantic_fidelity_winner_condition_from_llm",
    },
    "emotion_consistency": {
        "winner": "emotion_consistency_winner_A_B_tie",
        "A": "emotion_consistency_A_score_1_5",
        "B": "emotion_consistency_B_score_1_5",
        "llm": "emotion_consistency_winner_condition_from_llm",
    },
    "fluency": {
        "winner": "fluency_winner_A_B_tie",
        "A": "fluency_A_score_1_5",
        "B": "fluency_B_score_1_5",
        "llm": "fluency_winner_condition_from_llm",
    },
    "overall_preference": {
        "winner": "overall_preference_A_B_tie",
        "A": None,
        "B": None,
        "llm": "overall_preference_winner_condition_from_llm",
    },
}

INVALID_VALUES = {"-", "–", "—", "invalid", "제외", "불가", "평가불가"}


def read_table(path):
    path = Path(path)

    with open(path, "rb") as f:
        magic = f.read(4)

    if path.suffix.lower() in [".xlsx", ".xls"] or magic.startswith(b"PK"):
        return pd.read_excel(path, engine="openpyxl")

    for enc in ["utf-8-sig", "utf-8", "cp949", "euc-kr", "gb18030", "latin1"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass

    return pd.read_csv(path, encoding="utf-8-sig", engine="python", on_bad_lines="skip")


def write_xlsx(df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(path, index=False)


def norm(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def normalize_winner(x):
    x = norm(x)

    if x.upper() == "A":
        return "A"
    if x.upper() == "B":
        return "B"
    if x.lower() == "tie":
        return "tie"
    if x.lower() in INVALID_VALUES or x in INVALID_VALUES:
        return "invalid"

    return "invalid"


def map_to_condition(winner, a_condition, b_condition):
    w = normalize_winner(winner)

    if w == "A":
        return a_condition
    if w == "B":
        return b_condition
    if w == "tie":
        return "tie"

    return "invalid"


def to_score(x):
    x = norm(x)

    if x == "" or x.lower() in INVALID_VALUES or x in INVALID_VALUES:
        return np.nan

    return pd.to_numeric(x, errors="coerce")


def cohens_dz(diff):
    diff = np.asarray(diff, dtype=float)
    diff = diff[~np.isnan(diff)]

    if len(diff) < 2:
        return np.nan

    sd = np.std(diff, ddof=1)

    if sd == 0:
        return 0.0

    return float(np.mean(diff) / sd)


def paired_test(diff):
    diff = np.asarray(diff, dtype=float)
    diff = diff[~np.isnan(diff)]

    if len(diff) < 2:
        return np.nan, np.nan, np.nan, np.nan

    t_stat, t_p = stats.ttest_1samp(diff, 0.0)

    try:
        w_stat, w_p = stats.wilcoxon(diff)
    except Exception:
        w_stat, w_p = np.nan, np.nan

    return float(t_stat), float(t_p), float(w_stat), float(w_p)


def fleiss_kappa(label_groups, labels):
    if not label_groups:
        return np.nan

    groups = [g for g in label_groups if len(g) == 3]

    if not groups:
        return np.nan

    n_items = len(groups)
    n_raters = 3

    table = []

    for g in groups:
        table.append([g.count(label) for label in labels])

    table = np.asarray(table, dtype=float)

    p_j = table.sum(axis=0) / (n_items * n_raters)
    p_i = ((table ** 2).sum(axis=1) - n_raters) / (n_raters * (n_raters - 1))

    p_bar = p_i.mean()
    p_e = (p_j ** 2).sum()

    if p_e == 1:
        return np.nan

    return float((p_bar - p_e) / (1 - p_e))


def load_replacement(replacement_dir):
    replacement_dir = Path(replacement_dir)

    key_path = replacement_dir / "human_eval_replacement_answer_key.xlsx"
    if not key_path.exists():
        key_path = replacement_dir / "human_eval_replacement_answer_key.csv"

    if not key_path.exists():
        return pd.DataFrame()

    key = read_table(key_path)
    all_rows = []

    for annotator in ANNOTATORS:
        sheet_path = replacement_dir / annotator / f"{annotator}_replacement_sheet.xlsx"

        if not sheet_path.exists():
            sheet_path = replacement_dir / annotator / f"{annotator}_replacement_sheet.csv"

        if not sheet_path.exists():
            continue

        sheet = read_table(sheet_path)

        if len(sheet) == 0:
            continue

        key_sub = key[key["annotator"] == annotator].copy()

        merged = sheet.merge(
            key_sub,
            on=["eval_item_id", "sample_id"],
            how="left",
            suffixes=("", "_key"),
        )

        merged["annotator"] = annotator
        merged["source_part"] = "replacement"

        all_rows.append(merged)

    if not all_rows:
        return pd.DataFrame()

    return pd.concat(all_rows, ignore_index=True)


def prepare_final_df(valid_existing, replacement_df):
    valid_existing = valid_existing.copy()
    valid_existing["source_part"] = "existing_valid"

    if replacement_df is not None and len(replacement_df) > 0:
        df = pd.concat([valid_existing, replacement_df], ignore_index=True)
    else:
        df = valid_existing.copy()

    if "annotator" not in df.columns:
        raise ValueError("annotator 열이 없습니다.")

    if "A_condition" not in df.columns or "B_condition" not in df.columns:
        raise ValueError("A_condition 또는 B_condition 열이 없습니다.")

    if "is_shared_item" not in df.columns and "is_shared_item_key" in df.columns:
        df["is_shared_item"] = df["is_shared_item_key"]

    df["is_shared_item"] = df["is_shared_item"].astype(str).str.lower().isin(
        ["true", "1", "yes"]
    )

    for metric, cols in METRICS.items():
        winner_col = cols["winner"]

        df[f"{metric}_winner_norm"] = df[winner_col].apply(normalize_winner)

        df[f"{metric}_winner_condition_human"] = df.apply(
            lambda r: map_to_condition(
                r[winner_col],
                r["A_condition"],
                r["B_condition"],
            ),
            axis=1,
        )

        if cols["A"] is not None and cols["B"] is not None:
            a_score = f"{metric}_A_score_num"
            b_score = f"{metric}_B_score_num"

            df[a_score] = df[cols["A"]].apply(to_score)
            df[b_score] = df[cols["B"]].apply(to_score)

            df[f"{metric}_baseline_score"] = df.apply(
                lambda r: r[a_score] if r["A_condition"] == "baseline" else r[b_score],
                axis=1,
            )

            df[f"{metric}_audio_pred_score"] = df.apply(
                lambda r: r[a_score]
                if r["A_condition"] == "audio_pred_emotion_aware"
                else r[b_score],
                axis=1,
            )

            df[f"{metric}_score_diff"] = (
                df[f"{metric}_audio_pred_score"] - df[f"{metric}_baseline_score"]
            )

    return df


def summarize_winners(df):
    rows = []

    for metric in METRICS:
        col = f"{metric}_winner_condition_human"
        counts = df[col].value_counts(dropna=False).to_dict()

        audio = counts.get("audio_pred_emotion_aware", 0)
        baseline = counts.get("baseline", 0)
        tie = counts.get("tie", 0)
        invalid = counts.get("invalid", 0)

        valid_n = audio + baseline + tie
        non_tie = audio + baseline

        rows.append({
            "section": "human_pairwise_winner",
            "metric": metric,
            "n_total": len(df),
            "n_valid": valid_n,
            "audio_pred_wins": audio,
            "baseline_wins": baseline,
            "ties": tie,
            "invalid": invalid,
            "audio_win_rate_all_valid": audio / valid_n if valid_n else np.nan,
            "baseline_win_rate_all_valid": baseline / valid_n if valid_n else np.nan,
            "tie_rate_all_valid": tie / valid_n if valid_n else np.nan,
            "audio_win_rate_excluding_ties": audio / non_tie if non_tie else np.nan,
            "baseline_win_rate_excluding_ties": baseline / non_tie if non_tie else np.nan,
        })

    return rows


def summarize_scores(df):
    rows = []

    for metric, cols in METRICS.items():
        if cols["A"] is None:
            continue

        base_col = f"{metric}_baseline_score"
        audio_col = f"{metric}_audio_pred_score"
        diff_col = f"{metric}_score_diff"

        sub = df[[base_col, audio_col, diff_col]].dropna()

        t_stat, t_p, w_stat, w_p = paired_test(sub[diff_col])

        rows.append({
            "section": "human_score",
            "metric": metric,
            "n": len(sub),
            "baseline_mean": sub[base_col].mean(),
            "audio_pred_mean": sub[audio_col].mean(),
            "mean_diff_audio_minus_baseline": sub[diff_col].mean(),
            "paired_t": t_stat,
            "paired_t_p": t_p,
            "wilcoxon_stat": w_stat,
            "wilcoxon_p": w_p,
            "cohens_dz": cohens_dz(sub[diff_col]),
        })

    return rows


def summarize_llm_human_agreement(df):
    rows = []

    for metric, cols in METRICS.items():
        human_col = f"{metric}_winner_condition_human"
        llm_col = cols["llm"]

        if llm_col not in df.columns:
            continue

        sub = df[
            df[human_col].isin(["baseline", "audio_pred_emotion_aware", "tie"])
            & df[llm_col].isin(["baseline", "audio_pred_emotion_aware", "tie"])
        ].copy()

        agree = (sub[human_col] == sub[llm_col]).mean() if len(sub) else np.nan

        rows.append({
            "section": "llm_human_agreement",
            "metric": metric,
            "n": len(sub),
            "exact_agreement_rate": agree,
        })

    return rows


def summarize_agreement(df):
    labels = ["baseline", "audio_pred_emotion_aware", "tie", "invalid"]
    rows = []

    shared = df[df["is_shared_item"] == True].copy()

    for metric in METRICS:
        col = f"{metric}_winner_condition_human"

        label_groups = []
        complete_agreement = []

        for _, sub in shared.groupby("sample_id"):
            sub = sub.drop_duplicates(subset=["annotator"])

            if len(sub) < 2:
                continue

            labels_this = sub[col].fillna("invalid").tolist()
            labels_this = [x if x in labels else "invalid" for x in labels_this]

            label_groups.append(labels_this)
            complete_agreement.append(1 if len(set(labels_this)) == 1 else 0)

        groups_3 = [g for g in label_groups if len(g) == 3]

        rows.append({
            "metric": metric,
            "n_shared_items_with_2plus_raters": len(label_groups),
            "n_shared_items_with_3_raters": len(groups_3),
            "complete_agreement_rate": np.mean(complete_agreement)
            if complete_agreement
            else np.nan,
            "fleiss_kappa_3_raters": fleiss_kappa(groups_3, labels)
            if groups_3
            else np.nan,
        })

    return pd.DataFrame(rows)


def summarize_by_emotion(df):
    if "pred_emotion" not in df.columns:
        return pd.DataFrame()

    rows = []

    for emotion, sub in df.groupby("pred_emotion"):
        row = {
            "pred_emotion": emotion,
            "n": len(sub),
        }

        for metric in METRICS:
            col = f"{metric}_winner_condition_human"
            counts = sub[col].value_counts().to_dict()

            audio = counts.get("audio_pred_emotion_aware", 0)
            baseline = counts.get("baseline", 0)
            tie = counts.get("tie", 0)
            non_tie = audio + baseline

            row[f"{metric}_audio_wins"] = audio
            row[f"{metric}_baseline_wins"] = baseline
            row[f"{metric}_ties"] = tie
            row[f"{metric}_audio_win_rate_excl_ties"] = (
                audio / non_tie if non_tie else np.nan
            )

        rows.append(row)

    return pd.DataFrame(rows).sort_values("n", ascending=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--valid_existing", default="human_eval_valid_existing.xlsx")
    parser.add_argument("--replacement_dir", default="human_eval_replacement_package")
    parser.add_argument("--merged_output", default="human_eval_final_merged.xlsx")
    parser.add_argument("--summary_output", default="human_eval_final_summary.xlsx")
    parser.add_argument("--by_emotion_output", default="human_eval_final_by_emotion.xlsx")
    parser.add_argument("--agreement_output", default="human_eval_inter_annotator_agreement.xlsx")
    args = parser.parse_args()

    valid_existing = read_table(args.valid_existing)
    replacement_df = load_replacement(args.replacement_dir)

    final_df = prepare_final_df(valid_existing, replacement_df)

    write_xlsx(final_df, args.merged_output)

    summary_rows = []
    summary_rows.extend(summarize_winners(final_df))
    summary_rows.extend(summarize_scores(final_df))
    summary_rows.extend(summarize_llm_human_agreement(final_df))

    summary_df = pd.DataFrame(summary_rows)
    by_emotion_df = summarize_by_emotion(final_df)
    agreement_df = summarize_agreement(final_df)

    write_xlsx(summary_df, args.summary_output)
    write_xlsx(by_emotion_df, args.by_emotion_output)
    write_xlsx(agreement_df, args.agreement_output)

    print("\n========== 완료 ==========")
    print("최종 병합 파일:", args.merged_output)
    print("요약 파일:", args.summary_output)
    print("감정별 요약 파일:", args.by_emotion_output)
    print("평가자 간 일치도 파일:", args.agreement_output)
    print("\n최종 평가 기록 수:", len(final_df))

    print("\n===== Human Evaluation Summary =====")
    print(summary_df)

    print("\n===== Inter-Annotator Agreement =====")
    print(agreement_df)


if __name__ == "__main__":
    main()