# 01_prepare_eval_inputs.py
# -*- coding: utf-8 -*-

import re
import argparse
from pathlib import Path
import pandas as pd


EF_LIST = [
    "습니다", "습니까", "ㅂ니다", "ㅂ니까",
    "어요", "아요", "해요", "네요", "군요", "죠", "지요",
    "잖아요", "거든요", "는데요", "고요", "아서요", "어서요", "니까요",
    "잖아", "거든", "는데", "은데", "ㄴ데",
    "더라", "더라고", "더라고요",
    "니", "냐", "나", "까", "까요",
    "자", "세요", "십시오", "라", "어라", "아라",
    "구나", "구만", "군", "네",
    "다", "대", "래", "래요", "대요",
    "어", "아", "해", "지",
    "지만", "니까", "어서", "아서", "고", "며", "면서",
]


def normalize_text(text):
    text = str(text).replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def strip_punct(text):
    return re.sub(r"[.!?。！？~…\"'“”‘’)\]\}]+$", "", text.strip())


def extract_last_eojeol(text):
    text = strip_punct(normalize_text(text))
    if not text:
        return ""
    return text.split()[-1]


def extract_ef(text):
    last = extract_last_eojeol(text)

    if not last:
        return pd.Series({
            "last_eojeol": "",
            "ef_found": "",
            "ef_count": 0,
            "ef_match_method": "empty",
        })

    matched = []
    for ef in sorted(EF_LIST, key=len, reverse=True):
        if last.endswith(ef):
            matched.append(ef)

    matched = list(dict.fromkeys(matched))

    return pd.Series({
        "last_eojeol": last,
        "ef_found": "|".join(matched),
        "ef_count": len(matched),
        "ef_match_method": "rule_based" if matched else "none",
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="own_segments.csv")
    parser.add_argument("--output", default="own_eval_inputs.csv")
    parser.add_argument("--sample_size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_duration", type=float, default=0.5)
    parser.add_argument("--max_duration", type=float, default=15.0)
    parser.add_argument("--no_sample", action="store_true")
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    required = ["segment_id", "audio_id", "source_text", "duration_sec", "segment_audio_path"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    df["source_text"] = df["source_text"].apply(normalize_text)

    df = df[df["source_text"].str.len() > 0].copy()
    df = df[df["duration_sec"].between(args.min_duration, args.max_duration)].copy()

    df["sample_id"] = df["segment_id"]

    ef_df = df["source_text"].apply(extract_ef)
    df = pd.concat([df, ef_df], axis=1)

    if not args.no_sample and args.sample_size > 0 and len(df) > args.sample_size:
        df = df.sample(n=args.sample_size, random_state=args.seed).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    keep_cols = [
        "sample_id",
        "segment_id",
        "audio_id",
        "source_text",
        "start_time",
        "end_time",
        "start_sec",
        "end_sec",
        "duration_sec",
        "raw_audio_path",
        "segment_audio_path",
        "srt_path",
        "last_eojeol",
        "ef_found",
        "ef_count",
        "ef_match_method",
    ]

    keep_cols = [c for c in keep_cols if c in df.columns]
    out = df[keep_cols].copy()

    out.to_csv(args.output, index=False, encoding="utf-8-sig")

    print("\n========== Done ==========")
    print("input:", args.input)
    print("output:", args.output)
    print("rows:", len(out))
    print("EF matched rows:", (out["ef_count"] > 0).sum())
    print("\nTop EF:")
    print(out[out["ef_found"] != ""]["ef_found"].value_counts().head(20))
    print("\nPreview:")
    print(out[["sample_id", "source_text", "last_eojeol", "ef_found", "ef_count"]].head(10))


if __name__ == "__main__":
    main()