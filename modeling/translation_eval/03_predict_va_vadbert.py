# 03_predict_va_vadbert.py
# -*- coding: utf-8 -*-

"""
Step 3. Predict Valence/Arousal for English translations using VAD-BERT.

Input:
    translation_outputs.csv

Required columns:
    valence
    arousal
    baseline_translation
    emotion_aware_translation

Output:
    va_predictions_vadbert.csv

Important:
- The current KEMDy19-derived data appear to use a 1-5 V/A scale.
- R_VA is therefore computed with denominator 2 * (source_max - source_min).
"""

import argparse
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification


MODEL_NAME = "RobroKools/vad-bert"


def load_vad_model(model_name, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return tokenizer, model


def predict_vad(text, tokenizer, model, device):
    inputs = tokenizer(
        str(text),
        return_tensors="pt",
        truncation=True,
        max_length=256,
        padding=True
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        values = outputs.logits.squeeze().detach().cpu().numpy().tolist()

    if not isinstance(values, list) or len(values) < 3:
        raise ValueError(f"VAD 출력이 예상과 다릅니다: {values}")

    return float(values[0]), float(values[1]), float(values[2])


def convert_scale(x, method, source_min, source_max):
    """
    Convert VAD-BERT output to the same scale as source V/A.

    method:
    - raw: no conversion
    - 0_1_to_source: x in [0,1] -> [source_min, source_max]
    - minus1_1_to_source: x in [-1,1] -> [source_min, source_max]
    """
    x = float(x)

    if method == "raw":
        return x

    if method == "0_1_to_source":
        return source_min + x * (source_max - source_min)

    if method == "minus1_1_to_source":
        return source_min + ((x + 1.0) / 2.0) * (source_max - source_min)

    raise ValueError(f"지원하지 않는 scale method: {method}")


def compute_r_va(v_ref, a_ref, v_pred, a_pred, source_min, source_max):
    denom = 2.0 * (source_max - source_min)

    score = 1.0 - (
        abs(float(v_ref) - float(v_pred))
        + abs(float(a_ref) - float(a_pred))
    ) / denom

    return max(0.0, min(1.0, float(score)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="translation_outputs.csv")
    parser.add_argument("--output", default="va_predictions_vadbert.csv")
    parser.add_argument("--model_name", default=MODEL_NAME)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--scale",
        choices=["raw", "0_1_to_source", "minus1_1_to_source"],
        default="0_1_to_source"
    )
    parser.add_argument("--source_min", type=float, default=1.0)
    parser.add_argument("--source_max", type=float, default=5.0)
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    required_cols = [
        "valence",
        "arousal",
        "baseline_translation",
        "emotion_aware_translation"
    ]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"필수 컬럼이 없습니다: {col}")

    print("\n========== Source V/A summary ==========")
    print(df[["valence", "arousal"]].describe())
    print(f"\nR_VA source scale: {args.source_min} - {args.source_max}")

    tokenizer, model = load_vad_model(args.model_name, args.device)

    rows = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        v_ref = float(row["valence"])
        a_ref = float(row["arousal"])

        b_v_raw, b_a_raw, b_d_raw = predict_vad(
            row["baseline_translation"],
            tokenizer,
            model,
            args.device
        )

        e_v_raw, e_a_raw, e_d_raw = predict_vad(
            row["emotion_aware_translation"],
            tokenizer,
            model,
            args.device
        )

        b_v = convert_scale(b_v_raw, args.scale, args.source_min, args.source_max)
        b_a = convert_scale(b_a_raw, args.scale, args.source_min, args.source_max)

        e_v = convert_scale(e_v_raw, args.scale, args.source_min, args.source_max)
        e_a = convert_scale(e_a_raw, args.scale, args.source_min, args.source_max)

        b_rva = compute_r_va(v_ref, a_ref, b_v, b_a, args.source_min, args.source_max)
        e_rva = compute_r_va(v_ref, a_ref, e_v, e_a, args.source_min, args.source_max)

        new_row = row.to_dict()
        new_row.update({
            "baseline_vad_valence_raw": b_v_raw,
            "baseline_vad_arousal_raw": b_a_raw,
            "baseline_vad_dominance_raw": b_d_raw,
            "baseline_pred_valence": b_v,
            "baseline_pred_arousal": b_a,
            "baseline_R_VA": b_rva,

            "emotion_aware_vad_valence_raw": e_v_raw,
            "emotion_aware_vad_arousal_raw": e_a_raw,
            "emotion_aware_vad_dominance_raw": e_d_raw,
            "emotion_aware_pred_valence": e_v,
            "emotion_aware_pred_arousal": e_a,
            "emotion_aware_R_VA": e_rva,
        })

        rows.append(new_row)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.output, index=False, encoding="utf-8-sig")

    print("\n저장 완료:", args.output)
    print("샘플 수:", len(out_df))

    print("\n========== Raw VAD summary ==========")
    print(out_df[[
        "baseline_vad_valence_raw",
        "baseline_vad_arousal_raw",
        "emotion_aware_vad_valence_raw",
        "emotion_aware_vad_arousal_raw"
    ]].describe())

    print("\n========== R_VA summary ==========")
    print(out_df[["baseline_R_VA", "emotion_aware_R_VA"]].describe())


if __name__ == "__main__":
    main()