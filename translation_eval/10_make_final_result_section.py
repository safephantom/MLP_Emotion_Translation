# 10_make_final_result_section.py
# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd
import numpy as np


INPUT_XLSX = "final_report_tables.xlsx"
OUTPUT_MD = "10_final_result_section_KR.md"
OUTPUT_XLSX = "10_final_key_tables.xlsx"


def read_sheet(sheet):
    try:
        return pd.read_excel(INPUT_XLSX, sheet_name=sheet, engine="openpyxl")
    except Exception:
        return pd.DataFrame()


def fmt(x, nd=3):
    if pd.isna(x):
        return "NA"
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)


def get_row(df, metric, section=None):
    sub = df.copy()

    if "metric" in sub.columns:
        sub = sub[sub["metric"] == metric]

    if section is not None and "section" in sub.columns:
        sub = sub[sub["section"] == section]

    if len(sub) == 0:
        return None

    return sub.iloc[0]


def make_result_text(human_summary, agreement, llm_summary):
    sem_score = get_row(human_summary, "semantic_fidelity", "human_score")
    emo_score = get_row(human_summary, "emotion_consistency", "human_score")
    flu_score = get_row(human_summary, "fluency", "human_score")

    sem_win = get_row(human_summary, "semantic_fidelity", "human_pairwise_winner")
    emo_win = get_row(human_summary, "emotion_consistency", "human_pairwise_winner")
    flu_win = get_row(human_summary, "fluency", "human_pairwise_winner")
    overall_win = get_row(human_summary, "overall_preference", "human_pairwise_winner")

    llm_emo = get_row(llm_summary, "emotion_consistency")

    text = f"""# 최종 결과 해석

## 1. 전체 결과 요약

본 실험은 음성 기반 감정 정보가 한국어 발화의 영어 번역에서 감정적 뉘앙스와 화자의 태도 보존에 기여하는지를 검토하였다. 전체 결과는 audio-model emotion-aware 번역이 일반적인 번역 품질 전반을 일관되게 향상시키기보다는, 특히 감정 일관성(emotion consistency) 차원에서 선택적인 개선 효과를 보인다는 점을 보여준다.

## 2. 인간 평가 결과

인간 평가 결과, audio-model emotion-aware 조건은 감정 일관성에서 baseline보다 뚜렷한 우세를 보였다. A/B/tie 비교에서 감정 일관성 기준 audio-aware 조건의 tie 제외 승률은 {fmt(emo_win.get("audio_win_rate_excluding_ties") if emo_win is not None else np.nan)}로 나타났다. 또한 1–5점 척도 평가에서도 audio-aware 조건은 baseline보다 평균 {fmt(emo_score.get("mean_diff_audio_minus_baseline") if emo_score is not None else np.nan)}점 높게 평가되었으며, Wilcoxon 검정 결과도 유의하였다(p = {fmt(emo_score.get("wilcoxon_p") if emo_score is not None else np.nan, 6)}, Cohen's dz = {fmt(emo_score.get("cohens_dz") if emo_score is not None else np.nan)}).

반면 의미 충실도에서는 audio-aware 조건의 우세가 관찰되지 않았다. 의미 충실도 점수 차이는 {fmt(sem_score.get("mean_diff_audio_minus_baseline") if sem_score is not None else np.nan)}로, 방향상 baseline이 소폭 우세했으나 통계적으로 유의하지 않았다(p = {fmt(sem_score.get("wilcoxon_p") if sem_score is not None else np.nan, 6)}). 이는 감정 정보를 주입한 번역이 일부 경우 정서적 표현이나 어조를 강화하는 대신, 문자적 의미 보존에서는 약한 trade-off를 보일 수 있음을 시사한다.

유창성에서는 두 조건 간 차이가 거의 나타나지 않았다. 유창성 평균 차이는 {fmt(flu_score.get("mean_diff_audio_minus_baseline") if flu_score is not None else np.nan)}였으며, 통계적으로도 유의하지 않았다(p = {fmt(flu_score.get("wilcoxon_p") if flu_score is not None else np.nan, 6)}). 따라서 audio-aware 조건은 영어 번역의 자연스러움을 크게 훼손하지 않으면서 감정 일관성을 개선한 것으로 해석할 수 있다.

전체 선호도에서는 audio-aware 조건이 baseline보다 더 자주 선택되었다. tie를 제외한 전체 선호도 승률은 {fmt(overall_win.get("audio_win_rate_excluding_ties") if overall_win is not None else np.nan)}로 나타났다. 다만 이 효과는 감정 일관성에서보다 완만했으며, 이는 전체 선호도가 의미 충실도, 감정 일관성, 유창성의 영향을 동시에 받기 때문으로 해석된다.

## 3. LLM 평가와 인간 평가의 관계

LLM 기반 블라인드 평가 역시 감정 일관성 차원에서 audio-aware 조건을 더 선호하는 경향을 보였다. LLM judge 기준 감정 일관성의 tie 제외 audio-aware 승률은 {fmt(llm_emo.get("audio_win_rate_excluding_ties") if llm_emo is not None else np.nan)}였다. 인간 평가와 LLM 평가의 정확 일치율은 평가 차원별로 대략 중간 수준에 머물렀으며, 이는 LLM judge가 인간 판단을 완전히 대체하기보다는 보조적 평가 지표로 사용하는 것이 적절함을 보여준다.

## 4. 평가자 간 일치도

공동 평가 문항 25개를 기준으로 평가자 간 일치도를 확인한 결과, 전체 선호도에서 가장 높은 일치도가 나타났다. Fleiss' kappa는 전체 선호도에서 {fmt(agreement.loc[agreement["metric"] == "overall_preference", "fleiss_kappa_3_raters"].iloc[0]) if len(agreement) and "fleiss_kappa_3_raters" in agreement.columns else "NA"}로 나타났으며, 다른 세부 차원에서는 상대적으로 낮은 일치도를 보였다. 이는 번역의 감정적 뉘앙스와 의미 충실도 판단이 상당히 주관적인 평가 과제임을 반영한다.

## 5. 결론

종합하면, audio-model emotion-aware 번역은 일반적인 번역 품질을 전면적으로 개선하는 방법이라기보다는, 음성에 포함된 감정 정보와 화자의 상호작용적 태도를 번역문에 반영하는 데 효과적인 방법으로 해석된다. 특히 감정 일관성에서는 인간 평가에서 통계적으로 유의한 개선이 확인되었으나, 의미 충실도와 유창성에서는 뚜렷한 개선이 나타나지 않았다. 따라서 본 방법의 기여는 번역의 정확성 자체보다는 감정적 의미 보존과 발화 태도 전달에 있다고 볼 수 있다.
"""
    return text


def main():
    human_summary = read_sheet("human_summary")
    agreement = read_sheet("human_agreement")
    llm_summary = read_sheet("llm_blind_summary")
    auto_rva = read_sheet("automatic_RVA")
    examples = read_sheet("qualitative_examples")

    text = make_result_text(human_summary, agreement, llm_summary)
    Path(OUTPUT_MD).write_text(text, encoding="utf-8")

    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        human_summary.to_excel(writer, sheet_name="human_summary", index=False)
        agreement.to_excel(writer, sheet_name="human_agreement", index=False)
        llm_summary.to_excel(writer, sheet_name="llm_blind_summary", index=False)
        auto_rva.to_excel(writer, sheet_name="automatic_RVA", index=False)
        examples.to_excel(writer, sheet_name="qualitative_examples", index=False)

    print("\n========== 완료 ==========")
    print("결과 해석 초안:", OUTPUT_MD)
    print("핵심 표 파일:", OUTPUT_XLSX)


if __name__ == "__main__":
    main()