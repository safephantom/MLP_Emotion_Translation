# 04_evaluate_quality_deepseek_blind.py
# -*- coding: utf-8 -*-

import os
import re
import json
import time
import random
import argparse

import pandas as pd
from tqdm import tqdm
from openai import OpenAI


BASE_URL = "https://api.deepseek.com"


def get_client():
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError(
            "DEEPSEEK_API_KEY 환경변수를 설정하세요.\n"
            "$env:DEEPSEEK_API_KEY='sk-bed03e11ec1e451c90b1c6b9b7d6b523'"
        )
    return OpenAI(api_key=api_key, base_url=BASE_URL)


def extract_json(text):
    text = str(text).strip()
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise ValueError(f"JSON 객체를 찾을 수 없습니다:\n{text}")
        return json.loads(match.group(0))


def make_prompt(source, trans_a, trans_b):
    return f"""
Evaluate two English translations of one Korean source sentence.

The order of Translation A and Translation B is random.
Do not guess which system produced each translation.
Score each translation independently.

Score scale:
1 = very poor
2 = poor
3 = acceptable
4 = good
5 = excellent

Criteria:
- semantic_fidelity: meaning preservation
- emotion_consistency: emotional tone preservation
- fluency: natural English fluency

Korean source:
{source}

Translation A:
{trans_a}

Translation B:
{trans_b}

Return JSON only:
{{
  "A": {{
    "semantic_fidelity": 0,
    "emotion_consistency": 0,
    "fluency": 0
  }},
  "B": {{
    "semantic_fidelity": 0,
    "emotion_consistency": 0,
    "fluency": 0
  }}
}}
""".strip()


def call_judge(client, model, prompt, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=0,
                max_tokens=800,
                stream=False,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a strict Korean-English translation evaluator. "
                            "Return valid JSON only."
                        )
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                extra_body={
                    "thinking": {
                        "type": "disabled"
                    }
                }
            )
            return extract_json(response.choices[0].message.content)

        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            time.sleep(2 ** attempt)


def get_score(result, label, key):
    value = float(result[label][key])
    return max(1.0, min(5.0, value))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="va_predictions_vadbert.csv")
    parser.add_argument("--output", default="quality_eval_deepseek_blind.csv")
    parser.add_argument("--model", default="deepseek-v4-pro")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--save_every", type=int, default=20)
    args = parser.parse_args()

    random.seed(args.seed)

    df = pd.read_csv(args.input)

    required = [
        "sample_id",
        "source_text",
        "baseline_translation",
        "emotion_aware_translation"
    ]

    for col in required:
        if col not in df.columns:
            raise ValueError(f"필수 컬럼이 없습니다: {col}")

    if args.limit:
        df = df.iloc[:args.limit].copy()

    client = get_client()
    rows = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        source = str(row["source_text"])
        baseline = str(row["baseline_translation"])
        emotion_aware = str(row["emotion_aware_translation"])

        if random.random() < 0.5:
            trans_a = baseline
            trans_b = emotion_aware
            order = "A=baseline;B=emotion_aware"
        else:
            trans_a = emotion_aware
            trans_b = baseline
            order = "A=emotion_aware;B=baseline"

        prompt = make_prompt(source, trans_a, trans_b)
        result = call_judge(client, args.model, prompt)

        new_row = row.to_dict()
        new_row["blind_order"] = order
        new_row["llm_judge_model"] = args.model

        if order == "A=baseline;B=emotion_aware":
            base_label = "A"
            emo_label = "B"
        else:
            base_label = "B"
            emo_label = "A"

        new_row["baseline_semantic_fidelity"] = get_score(result, base_label, "semantic_fidelity")
        new_row["baseline_emotion_consistency"] = get_score(result, base_label, "emotion_consistency")
        new_row["baseline_fluency"] = get_score(result, base_label, "fluency")

        new_row["emotion_aware_semantic_fidelity"] = get_score(result, emo_label, "semantic_fidelity")
        new_row["emotion_aware_emotion_consistency"] = get_score(result, emo_label, "emotion_consistency")
        new_row["emotion_aware_fluency"] = get_score(result, emo_label, "fluency")

        rows.append(new_row)

        if len(rows) % args.save_every == 0:
            pd.DataFrame(rows).to_csv(args.output, index=False, encoding="utf-8-sig")
            print(f"[중간 저장] {args.output}, rows={len(rows)}")

    out = pd.DataFrame(rows)
    out.to_csv(args.output, index=False, encoding="utf-8-sig")

    print("\n저장 완료:", args.output)
    print("샘플 수:", len(out))
    print("\nA/B 顺序分布:")
    print(out["blind_order"].value_counts())

    print("\n评分摘要:")
    print(out[[
        "baseline_semantic_fidelity",
        "emotion_aware_semantic_fidelity",
        "baseline_emotion_consistency",
        "emotion_aware_emotion_consistency",
        "baseline_fluency",
        "emotion_aware_fluency"
    ]].describe())


if __name__ == "__main__":
    main()