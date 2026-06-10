# 03_generate_translation_audio_pred_deepseek.py
# -*- coding: utf-8 -*-

import os
import re
import json
import time
import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from openai import OpenAI


BASE_URL = "https://api.deepseek.com"


def get_client():
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError(
            "请先在 PowerShell 设置 API Key:\n"
            '$env:DEEPSEEK_API_KEY="sk-bed03e11ec1e451c90b1c6b9b7d6b523"'
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
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            raise ValueError(f"JSON parse failed:\n{text}")
        return json.loads(m.group(0))


def call_deepseek(client, model, prompt, max_retries=3):
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=0,
                max_tokens=512,
                stream=False,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a professional Korean-English translator. "
                            "Return valid JSON only."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                extra_body={"thinking": {"type": "disabled"}},
            )
            return extract_json(resp.choices[0].message.content)

        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            time.sleep(2 ** attempt)


def make_baseline_prompt(source_text):
    return f"""
Translate the Korean source into natural English.

Requirements:
- Preserve the original meaning accurately.
- Do not add explanations.
- Do not add emotions that are not present in the source.
- Return JSON only.

Korean source:
{source_text}

JSON format:
{{
  "translation": "..."
}}
""".strip()


def make_audio_pred_prompt(source_text, pred_emotion, pred_valence, pred_arousal, ef_found):
    return f"""
Translate the Korean source into natural English.

Use the following affective information predicted from the original audio and Korean ending-form features as guidance.

Predicted affective information:
- predicted emotion: {pred_emotion}
- predicted valence: {pred_valence}
- predicted arousal: {pred_arousal}
- Korean ending forms / EF: {ef_found}

Requirements:
- Preserve the original meaning accurately.
- Preserve the speaker's emotional nuance through word choice, modality, punctuation, and sentence style.
- Do not explicitly explain the emotion unless the source explicitly does so.
- Do not exaggerate the emotion.
- Do not add new factual information.
- Return JSON only.

Korean source:
{source_text}

JSON format:
{{
  "translation": "..."
}}
""".strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="own_eval_inputs_with_vae.csv")
    parser.add_argument("--output", default="translation_outputs_audio_pred.csv")
    parser.add_argument("--model", default="deepseek-v4-pro")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--save_every", type=int, default=20)
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    required = [
        "sample_id",
        "source_text",
        "ef_found",
        "pred_emotion",
        "pred_valence",
        "pred_arousal",
    ]

    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    if args.limit:
        df = df.iloc[:args.limit].copy()

    client = get_client()
    rows = []
    output_path = Path(args.output)

    for _, row in tqdm(df.iterrows(), total=len(df)):
        source_text = str(row["source_text"])
        ef_found = "" if pd.isna(row["ef_found"]) else str(row["ef_found"])

        baseline_prompt = make_baseline_prompt(source_text)
        audio_pred_prompt = make_audio_pred_prompt(
            source_text=source_text,
            pred_emotion=row["pred_emotion"],
            pred_valence=row["pred_valence"],
            pred_arousal=row["pred_arousal"],
            ef_found=ef_found,
        )

        baseline_result = call_deepseek(client, args.model, baseline_prompt)
        audio_pred_result = call_deepseek(client, args.model, audio_pred_prompt)

        new_row = row.to_dict()
        new_row["baseline_translation"] = baseline_result.get("translation", "")
        new_row["audio_pred_emotion_aware_translation"] = audio_pred_result.get("translation", "")
        new_row["translation_generation_model"] = args.model

        rows.append(new_row)

        if len(rows) % args.save_every == 0:
            pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8-sig")
            print(f"Saved intermediate file: {output_path}, rows={len(rows)}")

    out = pd.DataFrame(rows)
    out.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("\n========== Done ==========")
    print("input:", args.input)
    print("output:", args.output)
    print("rows:", len(out))
    print(out[[
        "sample_id",
        "source_text",
        "pred_emotion",
        "baseline_translation",
        "audio_pred_emotion_aware_translation",
    ]].head())


if __name__ == "__main__":
    main()