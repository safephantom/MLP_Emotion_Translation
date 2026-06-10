# 00_inspect_audio_model_project.py
# -*- coding: utf-8 -*-

import sys
from pathlib import Path
import torch


def print_section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def main():
    project_root = Path(__file__).resolve().parents[1]
    modeling_dir = project_root / "modeling"

    print_section("Path check")
    print("project_root:", project_root)
    print("modeling_dir:", modeling_dir)
    print("modeling exists:", modeling_dir.exists())

    if not modeling_dir.exists():
        raise FileNotFoundError(f"modeling folder not found: {modeling_dir}")

    sys.path.insert(0, str(modeling_dir))

    print_section("Files in modeling/")
    for p in sorted(modeling_dir.iterdir()):
        print(p.name)

    print_section("Important files")
    important = [
        "train_multimodal.py",
        "train_audio_only.py",
        "dataset.py",
        "kemdy19_multimodal_lstm.pth",
        "kemdy19_audio_only.pth",
        "merged_dataset_soft_fixed.csv",
        "dynamic_ef_weights_fixed.csv",
    ]

    for name in important:
        print(f"{name}: {(modeling_dir / name).exists()}")

    ckpt_path = modeling_dir / "kemdy19_multimodal_lstm.pth"

    print_section("Checkpoint inspection")
    print("checkpoint:", ckpt_path)

    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu")
        print("checkpoint type:", type(ckpt))

        if isinstance(ckpt, dict):
            print("\ncheckpoint keys:")
            for k in ckpt.keys():
                print(" -", k)

            if "model_state_dict" in ckpt:
                state = ckpt["model_state_dict"]
                print("\nUsing ckpt['model_state_dict']")
            elif "state_dict" in ckpt:
                state = ckpt["state_dict"]
                print("\nUsing ckpt['state_dict']")
            else:
                state = ckpt
                print("\nUsing checkpoint itself as state_dict")

            if isinstance(state, dict):
                print("\nstate_dict sample keys:")
                for i, (k, v) in enumerate(state.items()):
                    if i >= 50:
                        break
                    shape = tuple(v.shape) if hasattr(v, "shape") else "N/A"
                    print(f" - {k}: {shape}")
        else:
            print("checkpoint may be a full model object.")
    else:
        print("checkpoint not found.")

    print_section("Import dataset.py")
    try:
        import dataset
        print("dataset.py import: SUCCESS")
        for name in dir(dataset):
            if "Dataset" in name or "collate" in name or "Audio" in name or "EF" in name:
                print(" -", name)
    except Exception as e:
        print("dataset.py import: FAILED")
        print(repr(e))

    print_section("Import train_multimodal.py")
    try:
        import train_multimodal
        print("train_multimodal.py import: SUCCESS")
        for name in dir(train_multimodal):
            lower = name.lower()
            if (
                "model" in lower
                or "lstm" in lower
                or "multi" in lower
                or "emotion" in lower
                or "train" in lower
                or "dataset" in lower
            ):
                print(" -", name)
    except Exception as e:
        print("train_multimodal.py import: FAILED")
        print(repr(e))

    print_section("Import train_audio_only.py")
    try:
        import train_audio_only
        print("train_audio_only.py import: SUCCESS")
        for name in dir(train_audio_only):
            lower = name.lower()
            if (
                "model" in lower
                or "lstm" in lower
                or "audio" in lower
                or "emotion" in lower
                or "train" in lower
                or "dataset" in lower
            ):
                print(" -", name)
    except Exception as e:
        print("train_audio_only.py import: FAILED")
        print(repr(e))

    print_section("Done")
    print("把 checkpoint keys、state_dict sample keys、dataset.py 和 train_multimodal.py 的输出发我。")


if __name__ == "__main__":
    main()