# 00_prepare_own_segments_from_srt.py
# -*- coding: utf-8 -*-

import re
import csv
import argparse
import subprocess
from pathlib import Path

import pandas as pd

AUDIO_EXTS = [".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg"]


def read_text_with_fallback(path: Path) -> str:
    encodings = [
        "utf-8-sig", "utf-8",
        "cp949", "euc-kr",
        "utf-16", "utf-16-le", "utf-16-be",
        "utf-32", "utf-32-le", "utf-32-be",
    ]

    raw = path.read_bytes()

    for enc in encodings:
        try:
            text = raw.decode(enc)
            text = text.replace("\x00", "")
            if text.strip():
                return text
        except Exception:
            continue

    text = raw.decode("utf-8", errors="ignore").replace("\x00", "")
    if text.strip():
        print(f"[Warning] {path.name}: decoded with utf-8 errors='ignore'")
        return text

    raise RuntimeError(f"Cannot decode SRT file: {path}")


def srt_time_to_seconds(t: str) -> float:
    t = t.strip().replace(".", ",")
    m = re.match(r"(\d{1,2}):(\d{2}):(\d{2}),(\d{1,3})", t)
    if not m:
        raise ValueError(f"Invalid SRT time: {t}")

    h, mi, s, ms = m.groups()
    ms = ms.ljust(3, "0")[:3]

    return int(h) * 3600 + int(mi) * 60 + int(s) + int(ms) / 1000.0


def clean_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("\ufeff", "")
    text = text.replace("\x00", "")
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def parse_srt(path: Path):
    content = read_text_with_fallback(path)
    content = content.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")

    time_pattern = re.compile(
        r"(\d{1,2}:\d{2}:\d{2}[,.]\d{1,3})\s*(?:-->|–>|—>|->|→)\s*(\d{1,2}:\d{2}:\d{2}[,.]\d{1,3})"
    )

    lines = content.split("\n")
    segments = []
    i = 0

    while i < len(lines):
        line = lines[i].strip()
        m = time_pattern.search(line)

        if not m:
            i += 1
            continue

        start_time, end_time = m.group(1), m.group(2)
        text_lines = []
        i += 1

        while i < len(lines):
            next_line = lines[i].strip()

            if not next_line:
                break

            if time_pattern.search(next_line):
                i -= 1
                break

            if not re.fullmatch(r"\d+", next_line):
                text_lines.append(next_line)

            i += 1

        text = clean_text(" ".join(text_lines))

        if text:
            start_sec = srt_time_to_seconds(start_time)
            end_sec = srt_time_to_seconds(end_time)

            if end_sec > start_sec:
                segments.append({
                    "start_time": start_time,
                    "end_time": end_time,
                    "start_sec": start_sec,
                    "end_sec": end_sec,
                    "duration_sec": end_sec - start_sec,
                    "source_text": text,
                })

        i += 1

    return segments


def build_audio_index(audio_dir: Path):
    audio_index = {}

    for p in audio_dir.rglob("*"):
        if p.is_file() and not p.name.startswith("._") and p.suffix.lower() in AUDIO_EXTS:
            audio_index[p.stem] = p

    return audio_index


def find_matched_audio(audio_id: str, audio_index: dict):
    if audio_id in audio_index:
        return audio_index[audio_id]

    audio_id_norm = audio_id.lower().replace(" ", "").replace("-", "_")
    candidates = []

    for stem, path in audio_index.items():
        stem_norm = stem.lower().replace(" ", "").replace("-", "_")

        if (
            stem_norm == audio_id_norm
            or stem_norm.startswith(audio_id_norm)
            or audio_id_norm.startswith(stem_norm)
            or stem_norm.replace("_audio", "") == audio_id_norm
            or stem_norm.replace("_ko", "") == audio_id_norm.replace("_ko", "")
        ):
            candidates.append(path)

    if len(candidates) == 1:
        return candidates[0]

    if len(candidates) > 1:
        print(f"[Warning] Multiple audio candidates for {audio_id}: {[p.name for p in candidates]}")
        return candidates[0]

    return None


def cut_audio(input_audio, output_audio, start_sec, end_sec, ffmpeg_path, padding):
    output_audio.parent.mkdir(parents=True, exist_ok=True)

    start = max(0.0, start_sec - padding)
    duration = max(0.01, end_sec + padding - start)

    cmd = [
        ffmpeg_path,
        "-y",
        "-i", str(input_audio),
        "-ss", f"{start:.3f}",
        "-t", f"{duration:.3f}",
        "-ac", "1",
        "-ar", "16000",
        str(output_audio),
    ]

    r = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="ignore",
    )

    if r.returncode != 0:
        raise RuntimeError(r.stderr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_dir", default="../data_own/raw_audio")
    parser.add_argument("--srt_dir", default="../data_own/transcripts")
    parser.add_argument("--segment_audio_dir", default="../data_own/segment_audio")
    parser.add_argument("--output_csv", default="own_segments.csv")
    parser.add_argument("--padding", type=float, default=0.15)
    parser.add_argument("--min_duration", type=float, default=0.3)
    parser.add_argument("--no_cut", action="store_true")
    parser.add_argument("--ffmpeg_path", default="ffmpeg")
    args = parser.parse_args()

    audio_dir = Path(args.audio_dir)
    srt_dir = Path(args.srt_dir)
    segment_audio_dir = Path(args.segment_audio_dir)

    audio_index = build_audio_index(audio_dir)
    srt_files = [p for p in sorted(srt_dir.rglob("*.srt")) if not p.name.startswith("._")]

    print("\n========== Audio files ==========")
    print("audio_dir:", audio_dir)
    print("audio files:", len(audio_index))

    print("\n========== SRT files ==========")
    print("srt_dir:", srt_dir)
    print("srt files:", len(srt_files))

    rows = []
    missing_audio = []
    failed_srt = []
    failed_cut = []

    for srt_path in srt_files:
        audio_id = srt_path.stem
        raw_audio_path = find_matched_audio(audio_id, audio_index)

        print(f"\nProcessing SRT: {srt_path.name}")

        if raw_audio_path is None:
            missing_audio.append(audio_id)
            print(f"[Warning] No matched audio: {audio_id}")
            continue

        print(f"Matched audio: {raw_audio_path.name}")

        try:
            segments = parse_srt(srt_path)
        except Exception as e:
            failed_srt.append((srt_path.name, repr(e)))
            print(f"[Error] Failed to parse SRT: {srt_path.name}")
            print(repr(e))
            continue

        print(f"{audio_id}: {len(segments)} segments")

        valid_idx = 0

        for seg in segments:
            if seg["duration_sec"] < args.min_duration:
                continue

            valid_idx += 1
            segment_id = f"{audio_id}_seg{valid_idx:04d}"
            segment_audio_path = segment_audio_dir / f"{segment_id}.wav"

            if not args.no_cut:
                try:
                    cut_audio(
                        input_audio=raw_audio_path,
                        output_audio=segment_audio_path,
                        start_sec=seg["start_sec"],
                        end_sec=seg["end_sec"],
                        ffmpeg_path=args.ffmpeg_path,
                        padding=args.padding,
                    )
                except Exception as e:
                    failed_cut.append((segment_id, repr(e)))
                    print(f"[Error] Failed to cut: {segment_id}")
                    continue

            rows.append({
                "audio_id": audio_id,
                "segment_id": segment_id,
                "source_text": seg["source_text"],
                "start_time": seg["start_time"],
                "end_time": seg["end_time"],
                "start_sec": round(seg["start_sec"], 3),
                "end_sec": round(seg["end_sec"], 3),
                "duration_sec": round(seg["duration_sec"], 3),
                "raw_audio_path": str(raw_audio_path),
                "segment_audio_path": str(segment_audio_path),
                "srt_path": str(srt_path),
            })

    out = pd.DataFrame(rows)
    out.to_csv(args.output_csv, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)

    print("\n========== Done ==========")
    print("output_csv:", args.output_csv)
    print("total segments saved:", len(out))
    print("missing audio count:", len(missing_audio))
    print("failed SRT count:", len(failed_srt))
    print("failed cut count:", len(failed_cut))

    if len(out) > 0:
        print(out.head())


if __name__ == "__main__":
    main()