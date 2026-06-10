# 05_evaluate_quality_deepseek_blind.py
# -*- coding: utf-8 -*-

import os
import re
import json
import time
import random
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
            "PowerShell에서 API Key를 설정:\n"
            '$env:DEEPSEEK_API_KEY="sk-bed03e11ec1e451c90b1c6b9b7d6b523"'
        )

    return OpenAI(
        api_key=api_key,
        base_url=BASE_URL,
    )


def clean_json_text(text):
    text = str(text).strip()
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def extract_json(text):
    text = clean_json_text(text)

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
                max_tokens=1024,
                stream=False,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a strict and fair translation evaluator. "
                            "You must return valid JSON only."
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                extra_body={"thinking": {"type": "disabled"}},
            )

            return extract_json(resp.choices[0].message.content)

        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            time.sleep(2 ** attempt)


def make_eval_prompt(source_text, pred_emotion, pred_valence, pred_arousal, ef_found, trans_a, trans_b):
    return f"""
You are evaluating two English translations of the same Korean source sentence.

The source-side affective information below was predicted from the original Korean audio and Korean ending-form features.
Use it only as weak contextual guidance. Do not over-reward a translation just because it sounds more emotional. Meaning preservation is still the most important criterion.

Korean source:
{source_text}

Predicted source-side affective information:
- predicted emotion: {pred_emotion}
- predicted valence: {pred_valence}
- predicted arousal: {pred_arousal}
- Korean ending forms / EF: {ef_found}

Translation A:
{trans_a}

Translation B:
{trans_b}

Evaluate Translation A and Translation B on the following dimensions.

1. semantic_fidelity:
How accurately does the translation preserve the meaning of the Korean source?

2. emotion_consistency:
How well does the translation preserve the emotional tone, nuance, modality, and interactional stance of the source?

3. fluency:
How natural and fluent is the English translation?

4. overall_preference:
Which translation is better overall?

Scoring rules:
- Use integer scores from 1 to 5.
- 1 = very poor
- 2 = poor
- 3 = acceptable
- 4 = good
- 5 = excellent
- For pairwise winners, choose only "A", "B", or "tie".
- Be strict.
- Do not favor longer translations automatically.
- Do not favor emotionally exaggerated translations if they distort the meaning.
- Return JSON only.

JSON format:
{{
  "semantic_fidelity": {{
    "A_score": 1,
    "B_score": 1,
    "winner": "A"
  }},
  "emotion_consistency": {{
    "A_score": 1,
    "B_score": 1,
    "winner": "A"
  }},
  "fluency": {{
    "A_score": 1,
    "B_score": 1,
    "winner": "A"
  }},
  "overall_preference": {{
    "winner": "A"
  }},
  "brief_reason": "one short sentence explaining the main difference"
}}
""".strip()


def safe_get_score(result, category, side):
    try:
        return int(result[category][f"{side}_score"])
    except Exception:
        return None


def safe_get_winner(result, category):
    try:
        w = str(result[category]["winner"]).strip()
        if w not in ["A", "B", "tie"]:
            return "tie"
        return w
    except Exception:
        return "tie"


def map_winner_to_condition(winner, a_condition, b_condition):
    if winner == "A":
        return a_condition
    elif winner == "B":
        return b_condition
    else:
        return "tie"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="va_predictions_audio_pred.csv")
    parser.add_argument("--output", default="quality_eval_audio_pred_blind.csv")
    parser.add_argument("--model", default="deepseek-v4-pro")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_every", type=int, default=20)
    args = parser.parse_args()

    random.seed(args.seed)

    df = pd.read_csv(args.input)

    required = [
        "sample_id",
        "source_text",
        "pred_emotion",
        "pred_valence",
        "pred_arousal",
        "ef_found",
        "baseline_translation",
        "audio_pred_emotion_aware_translation",
    ]

    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    if args.limit is not None:
        df = df.iloc[:args.limit].copy()

    client = get_client()
    output_path = Path(args.output)

    rows = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        baseline_translation = str(row["baseline_translation"])
        audio_translation = str(row["audio_pred_emotion_aware_translation"])

        # blind randomization
        if random.random() < 0.5:
            trans_a = baseline_translation
            trans_b = audio_translation
            a_condition = "baseline"
            b_condition = "audio_pred_emotion_aware"
        else:
            trans_a = audio_translation
            trans_b = baseline_translation
            a_condition = "audio_pred_emotion_aware"
            b_condition = "baseline"

        ef_found = "" if pd.isna(row["ef_found"]) else str(row["ef_found"])

        prompt = make_eval_prompt(
            source_text=row["source_text"],
            pred_emotion=row["pred_emotion"],
            pred_valence=row["pred_valence"],
            pred_arousal=row["pred_arousal"],
            ef_found=ef_found,
            trans_a=trans_a,
            trans_b=trans_b,
        )

        try:
            result = call_deepseek(client, args.model, prompt)
            error = ""
        except Exception as e:
            result = {}
            error = repr(e)

        sem_winner_ab = safe_get_winner(result, "semantic_fidelity")
        emo_winner_ab = safe_get_winner(result, "emotion_consistency")
        flu_winner_ab = safe_get_winner(result, "fluency")
        overall_winner_ab = safe_get_winner(result, "overall_preference")

        new_row = row.to_dict()

        new_row["A_condition"] = a_condition
        new_row["B_condition"] = b_condition
        new_row["translation_A"] = trans_a
        new_row["translation_B"] = trans_b

        new_row["A_semantic_fidelity_score"] = safe_get_score(result, "semantic_fidelity", "A")
        new_row["B_semantic_fidelity_score"] = safe_get_score(result, "semantic_fidelity", "B")
        new_row["semantic_fidelity_winner_AB"] = sem_winner_ab
        new_row["semantic_fidelity_winner_condition"] = map_winner_to_condition(
            sem_winner_ab, a_condition, b_condition
        )

        new_row["A_emotion_consistency_score"] = safe_get_score(result, "emotion_consistency", "A")
        new_row["B_emotion_consistency_score"] = safe_get_score(result, "emotion_consistency", "B")
        new_row["emotion_consistency_winner_AB"] = emo_winner_ab
        new_row["emotion_consistency_winner_condition"] = map_winner_to_condition(
            emo_winner_ab, a_condition, b_condition
        )

        new_row["A_fluency_score"] = safe_get_score(result, "fluency", "A")
        new_row["B_fluency_score"] = safe_get_score(result, "fluency", "B")
        new_row["fluency_winner_AB"] = flu_winner_ab
        new_row["fluency_winner_condition"] = map_winner_to_condition(
            flu_winner_ab, a_condition, b_condition
        )

        new_row["overall_preference_winner_AB"] = overall_winner_ab
        new_row["overall_preference_winner_condition"] = map_winner_to_condition(
            overall_winner_ab, a_condition, b_condition
        )

        new_row["brief_reason"] = result.get("brief_reason", "")
        new_row["judge_model"] = args.model
        new_row["judge_error"] = error

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

    print("\nWinner counts:")
    for col in [
        "semantic_fidelity_winner_condition",
        "emotion_consistency_winner_condition",
        "fluency_winner_condition",
        "overall_preference_winner_condition",
    ]:
        print("\n", col)
        print(out[col].value_counts(dropna=False))

    print("\nError count:")
    print((out["judge_error"].fillna("") != "").sum())

    print("\nPreview:")
    print(out[[
        "sample_id",
        "A_condition",
        "B_condition",
        "semantic_fidelity_winner_condition",
        "emotion_consistency_winner_condition",
        "fluency_winner_condition",
        "overall_preference_winner_condition",
        "brief_reason",
    ]].head(10))


if __name__ == "__main__":
    main()