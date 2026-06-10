# 04_predict_va_vadbert.py
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification


MODEL_NAME = "RobroKools/vad-bert"


def load_vad_model(device):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()
    return tokenizer, model


def predict_va(text, tokenizer, model, device):
    text = "" if pd.isna(text) else str(text)

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=256,
        padding=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        preds = outputs.logits.squeeze().detach().cpu().numpy()

    valence = float(preds[0])
    arousal = float(preds[1])
    return valence, arousal


def zscore(x):
    x = np.asarray(x, dtype=float)
    return (x - np.nanmean(x)) / (np.nanstd(x) + 1e-8)


def compute_r_va(source_v, source_a, trans_v, trans_a):
    dist = np.sqrt((source_v - trans_v) ** 2 + (source_a - trans_a) ** 2)
    return 1.0 / (1.0 + dist)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="translation_outputs_audio_pred.csv")
    parser.add_argument("--output", default="va_predictions_audio_pred.csv")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    df = pd.read_csv(args.input)

    required = [
        "sample_id",
        "source_text",
        "pred_valence",
        "pred_arousal",
        "baseline_translation",
        "audio_pred_emotion_aware_translation",
    ]

    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    tokenizer, model = load_vad_model(device)

    baseline_vs, baseline_as = [], []
    audio_pred_vs, audio_pred_as = [], []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        b_v, b_a = predict_va(row["baseline_translation"], tokenizer, model, device)
        t_v, t_a = predict_va(row["audio_pred_emotion_aware_translation"], tokenizer, model, device)

        baseline_vs.append(b_v)
        baseline_as.append(b_a)
        audio_pred_vs.append(t_v)
        audio_pred_as.append(t_a)

    df["baseline_translation_valence_raw"] = baseline_vs
    df["baseline_translation_arousal_raw"] = baseline_as
    df["audio_pred_translation_valence_raw"] = audio_pred_vs
    df["audio_pred_translation_arousal_raw"] = audio_pred_as

    # raw R_VA: only for reference, because source and target models are on different scales
    df["baseline_R_VA_raw"] = [
        compute_r_va(sv, sa, tv, ta)
        for sv, sa, tv, ta in zip(
            df["pred_valence"],
            df["pred_arousal"],
            df["baseline_translation_valence_raw"],
            df["baseline_translation_arousal_raw"],
        )
    ]

    df["audio_pred_R_VA_raw"] = [
        compute_r_va(sv, sa, tv, ta)
        for sv, sa, tv, ta in zip(
            df["pred_valence"],
            df["pred_arousal"],
            df["audio_pred_translation_valence_raw"],
            df["audio_pred_translation_arousal_raw"],
        )
    ]

    df["R_VA_raw_diff"] = df["audio_pred_R_VA_raw"] - df["baseline_R_VA_raw"]

    # z-score alignment
    # source V/A: standardized within source-side Audio+EF predictions
    df["source_valence_z"] = zscore(df["pred_valence"])
    df["source_arousal_z"] = zscore(df["pred_arousal"])

    # translation V/A: baseline and target are standardized together
    all_trans_v = np.concatenate([
        df["baseline_translation_valence_raw"].values,
        df["audio_pred_translation_valence_raw"].values,
    ])

    all_trans_a = np.concatenate([
        df["baseline_translation_arousal_raw"].values,
        df["audio_pred_translation_arousal_raw"].values,
    ])

    trans_v_mean = np.nanmean(all_trans_v)
    trans_v_std = np.nanstd(all_trans_v) + 1e-8
    trans_a_mean = np.nanmean(all_trans_a)
    trans_a_std = np.nanstd(all_trans_a) + 1e-8

    df["baseline_translation_valence_z"] = (
        df["baseline_translation_valence_raw"] - trans_v_mean
    ) / trans_v_std

    df["audio_pred_translation_valence_z"] = (
        df["audio_pred_translation_valence_raw"] - trans_v_mean
    ) / trans_v_std

    df["baseline_translation_arousal_z"] = (
        df["baseline_translation_arousal_raw"] - trans_a_mean
    ) / trans_a_std

    df["audio_pred_translation_arousal_z"] = (
        df["audio_pred_translation_arousal_raw"] - trans_a_mean
    ) / trans_a_std

    # main R_VA: z-score aligned
    df["baseline_R_VA"] = [
        compute_r_va(sv, sa, tv, ta)
        for sv, sa, tv, ta in zip(
            df["source_valence_z"],
            df["source_arousal_z"],
            df["baseline_translation_valence_z"],
            df["baseline_translation_arousal_z"],
        )
    ]

    df["audio_pred_R_VA"] = [
        compute_r_va(sv, sa, tv, ta)
        for sv, sa, tv, ta in zip(
            df["source_valence_z"],
            df["source_arousal_z"],
            df["audio_pred_translation_valence_z"],
            df["audio_pred_translation_arousal_z"],
        )
    ]

    df["R_VA_diff"] = df["audio_pred_R_VA"] - df["baseline_R_VA"]

    df.to_csv(args.output, index=False, encoding="utf-8-sig")

    print("\n========== Done ==========")
    print("input:", args.input)
    print("output:", args.output)
    print("rows:", len(df))

    print("\nRaw R_VA summary:")
    print(df[["baseline_R_VA_raw", "audio_pred_R_VA_raw", "R_VA_raw_diff"]].describe())

    print("\nZ-score aligned R_VA summary:")
    print(df[["baseline_R_VA", "audio_pred_R_VA", "R_VA_diff"]].describe())

    print("\nMean comparison:")
    print("baseline_R_VA mean:", df["baseline_R_VA"].mean())
    print("audio_pred_R_VA mean:", df["audio_pred_R_VA"].mean())
    print("diff mean:", df["R_VA_diff"].mean())

    print("\nPreview:")
    print(df[[
        "sample_id",
        "pred_emotion",
        "pred_valence",
        "pred_arousal",
        "baseline_R_VA",
        "audio_pred_R_VA",
        "R_VA_diff",
    ]].head(10))


if __name__ == "__main__":
    main()