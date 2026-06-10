# 02_predict_audio_ef_vae.py
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import librosa
from tqdm import tqdm


EMOTION_LABELS = ["anger", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"]


class AudioEFEmotionModel(nn.Module):
    def __init__(self, audio_input_dim=43, ef_dim=7, hidden_dim=128, num_layers=2, dropout=0.41):
        super().__init__()

        self.audio_lstm = nn.LSTM(
            input_size=audio_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )

        fused_dim = hidden_dim * 2 + ef_dim

        self.emotion_head = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 7),
        )

        self.valence_head = nn.Sequential(
            nn.Linear(fused_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

        self.arousal_head = nn.Sequential(
            nn.Linear(fused_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, audio_x, ef_x):
        # audio_x: [B, T, 43]
        lstm_out, _ = self.audio_lstm(audio_x)

        # mean pooling over time
        audio_repr = lstm_out.mean(dim=1)  # [B, 256]

        fused = torch.cat([audio_repr, ef_x], dim=1)  # [B, 263]

        emotion_logits = self.emotion_head(fused)
        valence = self.valence_head(fused).squeeze(-1)
        arousal = self.arousal_head(fused).squeeze(-1)

        return emotion_logits, valence, arousal


def extract_audio_features(audio_path, sr=16000):
    """
    Output shape: [T, 43]
    43 dims = MFCC13 + delta13 + delta2_13 + zcr + rms + centroid + bandwidth
    """
    y, sr = librosa.load(audio_path, sr=sr, mono=True)

    if len(y) < sr * 0.1:
        y = np.pad(y, (0, int(sr * 0.1) - len(y)))

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)

    feats = np.vstack([mfcc, delta, delta2, zcr, rms, centroid, bandwidth]).T

    if feats.shape[1] != 43:
        raise ValueError(f"Feature dim must be 43, got {feats.shape[1]}")

    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

    # simple utterance-level normalization
    mean = feats.mean(axis=0, keepdims=True)
    std = feats.std(axis=0, keepdims=True) + 1e-6
    feats = (feats - mean) / std

    return feats.astype(np.float32)


def load_ef_table(path):
    """
    dynamic_ef_weights_fixed.csv expected columns may include:
    ending / ef / endings_found + anger/disgust/fear/happiness/neutral/sadness/surprise
    """
    path = Path(path)

    if not path.exists():
        print(f"[Warning] EF table not found: {path}. Use uniform EF vector.")
        return {}

    df = pd.read_csv(path)

    ef_col = None
    for c in ["ending", "ef", "ending_form", "endings_found", "EF", "종결어미"]:
        if c in df.columns:
            ef_col = c
            break

    if ef_col is None:
        ef_col = df.columns[0]

    emotion_cols = [c for c in EMOTION_LABELS if c in df.columns]

    if len(emotion_cols) != 7:
        print("[Warning] EF table emotion columns not complete. Use uniform EF vector.")
        return {}

    table = {}

    for _, row in df.iterrows():
        key = str(row[ef_col]).strip()
        if not key or key.lower() == "nan":
            continue

        vec = row[emotion_cols].astype(float).values
        s = vec.sum()
        if s > 0:
            vec = vec / s
        else:
            vec = np.ones(7) / 7

        table[key] = vec.astype(np.float32)

    print(f"Loaded EF table: {len(table)} entries from {path}")
    return table


def ef_to_vector(ef_found, ef_table):
    """
    ef_found example:
        '니까|까'
        '잖아요|아요'
        ''
    If multiple EFs exist, average their vectors.
    """
    if not ef_table:
        return np.ones(7, dtype=np.float32) / 7

    if pd.isna(ef_found) or str(ef_found).strip() == "":
        return np.ones(7, dtype=np.float32) / 7

    efs = [x.strip() for x in str(ef_found).split("|") if x.strip()]
    vecs = []

    for ef in efs:
        if ef in ef_table:
            vecs.append(ef_table[ef])

    if not vecs:
        return np.ones(7, dtype=np.float32) / 7

    vec = np.mean(vecs, axis=0)
    vec = vec / (vec.sum() + 1e-8)

    return vec.astype(np.float32)


def load_model(checkpoint_path, device):
    model = AudioEFEmotionModel()
    state = torch.load(checkpoint_path, map_location=device)

    if isinstance(state, OrderedDict) or isinstance(state, dict):
        model.load_state_dict(state, strict=True)
    else:
        raise ValueError("Unsupported checkpoint format.")

    model.to(device)
    model.eval()

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="own_eval_inputs.csv")
    parser.add_argument("--output", default="own_eval_inputs_with_vae.csv")
    parser.add_argument("--checkpoint", default="../modeling/kemdy19_multimodal_lstm.pth")
    parser.add_argument("--ef_table", default="../modeling/dynamic_ef_weights_fixed.csv")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sr", type=int, default=16000)
    args = parser.parse_args()

    device = torch.device(args.device)

    df = pd.read_csv(args.input)

    required = ["sample_id", "source_text", "segment_audio_path", "ef_found"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    model = load_model(args.checkpoint, device)
    ef_table = load_ef_table(args.ef_table)

    pred_emotions = []
    pred_valences = []
    pred_arousals = []
    pred_probs_all = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        audio_path = Path(row["segment_audio_path"])

        if not audio_path.exists():
            pred_emotions.append("missing_audio")
            pred_valences.append(np.nan)
            pred_arousals.append(np.nan)
            pred_probs_all.append("")
            continue

        audio_feat = extract_audio_features(audio_path, sr=args.sr)
        ef_vec = ef_to_vector(row["ef_found"], ef_table)

        audio_tensor = torch.tensor(audio_feat, dtype=torch.float32).unsqueeze(0).to(device)
        ef_tensor = torch.tensor(ef_vec, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            emotion_logits, valence, arousal = model(audio_tensor, ef_tensor)
            probs = torch.softmax(emotion_logits, dim=-1).cpu().numpy()[0]

        pred_idx = int(np.argmax(probs))
        pred_emotion = EMOTION_LABELS[pred_idx]

        pred_emotions.append(pred_emotion)
        pred_valences.append(float(valence.cpu().item()))
        pred_arousals.append(float(arousal.cpu().item()))
        pred_probs_all.append("|".join([f"{x:.4f}" for x in probs]))

    df["pred_emotion"] = pred_emotions
    df["pred_valence"] = pred_valences
    df["pred_arousal"] = pred_arousals
    df["pred_emotion_probs"] = pred_probs_all

    df.to_csv(args.output, index=False, encoding="utf-8-sig")

    print("\n========== Done ==========")
    print("input:", args.input)
    print("output:", args.output)
    print("rows:", len(df))
    print("\nPred emotion distribution:")
    print(df["pred_emotion"].value_counts())
    print("\nPreview:")
    print(df[["sample_id", "source_text", "ef_found", "pred_emotion", "pred_valence", "pred_arousal"]].head(10))


if __name__ == "__main__":
    main()