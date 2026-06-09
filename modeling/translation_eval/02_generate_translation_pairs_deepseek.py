# 02_generate_translation_pairs_deepseek.py
# -*- coding: utf-8 -*-

"""
Step 2. Generate two Korean-English translations using DeepSeek.

Experimental design:
1. baseline_translation:
   - Only the Korean source text is provided.
2. emotion_aware_translation:
   - Korean source text + emotion + valence + arousal + EF information are provided.

Purpose:
- Compare whether explicit affective information improves emotion-preserving translation.
- Translation generation model is separated from evaluation model.
- DeepSeek is used for generation, while Claude is used later only for evaluation.

Input:
    translation_inputs.csv

Required columns:
    sample_id
    source_text
    emotion
    valence
    arousal
    ef_info

Output:
    translation_outputs.csv
"""

import os
import re
import json
import time
import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from openai import OpenAI


DEEPSEEK_BASE_URL = "https://api.deepseek.com"


def get_client():
    """
    Create DeepSeek client using OpenAI-compatible API.
    API key should be set in PowerShell:
        $env:DEEPSEEK_API_KEY="your_real_deepseek_api_key"
    """
    api_key = os.getenv("DEEPSEEK_API_KEY")

    if not api_key:
        raise RuntimeError(
            "DEEPSEEK_API_KEY 환경변수를 설정하세요.\n"
            "PowerShell 예:\n"
            "$env:DEEPSEEK_API_KEY='sk-bed03e11ec1e451c90b1c6b9b7d6b523'"
        )

    return OpenAI(
        api_key=api_key,
        base_url=DEEPSEEK_BASE_URL
    )


def extract_json(text: str) -> dict:
    """
    Robustly extract JSON from model output.
    """
    text = str(text).strip()

    # Remove markdown fences if the model returns ```json ... ```
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fallback: extract the first JSON-looking object
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"JSON 객체를 찾을 수 없습니다:\n{text}")

    return json.loads(match.group(0))


def call_deepseek_json(client, model, prompt, max_tokens=512, max_retries=4):
    """
    Call DeepSeek and parse JSON output.

    For this translation task, thinking mode is disabled because:
    1. The task is not complex reasoning.
    2. We need stable JSON output.
    3. Batch translation should be efficient and reproducible.
    """
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=0,
                max_tokens=max_tokens,
                stream=False,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a professional Korean-English translator. "
                            "Return valid JSON only. Do not include explanations, "
                            "comments, markdown, or additional text."
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

            text = response.choices[0].message.content
            return extract_json(text)

        except Exception as e:
            if attempt == max_retries - 1:
                raise e

            wait_time = 2 ** attempt
            print(f"[Retry] API 호출 실패. {wait_time}초 후 재시도: {e}")
            time.sleep(wait_time)


def make_baseline_prompt(source_text: str) -> str:
    """
    Baseline condition:
    only the source text is given.
    No explicit emotion, valence, arousal, or EF information is provided.
    """
    return f"""
Translate the following Korean utterance into natural English.

Requirements:
- Preserve the propositional meaning accurately.
- Do not add explanations.
- Do not add emotional descriptions that are not explicitly present in the source.
- Do not exaggerate the speaker's attitude.
- Output only the English translation.

Korean source:
{source_text}

Return JSON only:
{{
  "translation": "..."
}}
""".strip()


def make_emotion_aware_prompt(
    source_text: str,
    emotion: str,
    valence,
    arousal,
    ef_info: str
) -> str:
    """
    Emotion-aware condition:
    source text + source-side affective information are provided.
    """
    return f"""
Translate the following Korean utterance into natural English.
In addition to preserving the propositional meaning, preserve the speaker's emotional nuance.

Source-side affective information:
- emotion label: {emotion}
- valence: {valence}
- arousal: {arousal}
- Korean ending forms / EF: {ef_info}

Requirements:
- Preserve emotional tone through word choice, modality, punctuation, and sentence style.
- Do not explicitly explain the emotion unless the Korean source explicitly does so.
- Do not exaggerate the emotion.
- Do not add new information that is not present in the source.
- Output only the English translation.

Korean source:
{source_text}

Return JSON only:
{{
  "translation": "..."
}}
""".strip()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default="translation_inputs.csv",
        help="Input CSV file produced by Step 1"
    )

    parser.add_argument(
        "--output",
        default="translation_outputs.csv",
        help="Output CSV file"
    )

    parser.add_argument(
        "--model",
        default="deepseek-v4-flash",
        help="DeepSeek model name, e.g., deepseek-v4-flash or deepseek-v4-pro"
    )

    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start index for partial generation"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Number of samples to process. Use this for test runs."
    )

    parser.add_argument(
        "--save_every",
        type=int,
        default=20,
        help="Save intermediate output every N rows"
    )

    args = parser.parse_args()

    df = pd.read_csv(args.input)

    required_cols = [
        "sample_id",
        "source_text",
        "emotion",
        "valence",
        "arousal",
        "ef_info"
    ]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"필수 컬럼이 없습니다: {col}")

    if args.limit is not None:
        df = df.iloc[args.start: args.start + args.limit].copy()
    else:
        df = df.iloc[args.start:].copy()

    print("\n========== DeepSeek translation generation ==========")
    print("Input file:", args.input)
    print("Output file:", args.output)
    print("Model:", args.model)
    print("Samples to process:", len(df))

    client = get_client()

    rows = []
    output_path = Path(args.output)

    for _, row in tqdm(df.iterrows(), total=len(df)):
        source_text = str(row["source_text"])
        emotion = str(row["emotion"])
        valence = row["valence"]
        arousal = row["arousal"]
        ef_info = str(row["ef_info"]) if pd.notna(row["ef_info"]) else ""

        baseline_result = call_deepseek_json(
            client=client,
            model=args.model,
            prompt=make_baseline_prompt(source_text)
        )

        emotion_result = call_deepseek_json(
            client=client,
            model=args.model,
            prompt=make_emotion_aware_prompt(
                source_text=source_text,
                emotion=emotion,
                valence=valence,
                arousal=arousal,
                ef_info=ef_info
            )
        )

        baseline_translation = baseline_result.get("translation", "")
        emotion_aware_translation = emotion_result.get("translation", "")

        if not baseline_translation:
            print(f"[Warning] Empty baseline translation: {row['sample_id']}")

        if not emotion_aware_translation:
            print(f"[Warning] Empty emotion-aware translation: {row['sample_id']}")

        new_row = row.to_dict()
        new_row["baseline_translation"] = baseline_translation
        new_row["emotion_aware_translation"] = emotion_aware_translation
        new_row["translation_generation_model"] = args.model
        new_row["translation_generation_api"] = "DeepSeek"

        rows.append(new_row)

        if len(rows) % args.save_every == 0:
            temp_df = pd.DataFrame(rows)
            temp_df.to_csv(output_path, index=False, encoding="utf-8-sig")
            print(f"[중간 저장] {output_path}, rows={len(temp_df)}")

    out_df = pd.DataFrame(rows)
    out_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("\n저장 완료:", output_path)
    print("샘플 수:", len(out_df))

    print("\n========== 앞 5행 ==========")
    print(out_df[[
        "sample_id",
        "source_text",
        "emotion",
        "baseline_translation",
        "emotion_aware_translation"
    ]].head())


if __name__ == "__main__":
    main()