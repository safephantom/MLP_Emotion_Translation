## MLP_Emotion_Translation

2026-1 <기계학습과 딥러닝> 수업의 기말 프로젝트 결과물입니다.

# Emotion-Aware Korean Translation Program

This project develops an emotion-aware Korean-to-English translation framework that integrates acoustic emotion cues and Korean sentence-final ending information (EF) into the translation pipeline.

The main goal is to test whether a translation system can go beyond semantic-only translation and better preserve:

* emotional nuance,
* speaker attitude,
* interactional stance,
* and affective context.

The project compares a text-only baseline translation with an audio/EF-informed emotion-aware translation.

---

## Overview

Korean spoken utterances often encode emotion not only through lexical content, but also through prosody, speech intensity, and sentence-final endings.
This project uses an emotion recognition model trained on KEMDy19 to predict affective signals and injects them into the translation process.

```text
Korean speech + Korean text + EF information
        ↓
Audio/EF-based emotion prediction
        ↓
Predicted emotion / valence / arousal
        ↓
Emotion-aware translation prompting
        ↓
English translation
        ↓
Automatic + LLM + human evaluation
```

---

## Quick Start

Clone the repository and install dependencies:

```bash
git clone https://github.com/YOUR_USERNAME/MLP_Emotion_Translation.git
cd MLP_Emotion_Translation

pip install pandas numpy scipy openpyxl torch transformers librosa soundfile tqdm openai
```

Set the DeepSeek API key before running translation or LLM-judge scripts:

```bash
export DEEPSEEK_API_KEY="your_api_key_here"
```

For PowerShell:

```powershell
$env:DEEPSEEK_API_KEY="your_api_key_here"
```

Run the main translation evaluation pipeline step by step:

```bash
python translation_eval/01_prepare_eval_inputs.py
python translation_eval/02_predict_audio_ef_vae.py
python translation_eval/03_generate_translation_audio_pred_deepseek.py
python translation_eval/04_predict_va_vadbert.py
python translation_eval/05_evaluate_quality_deepseek_blind.py
python translation_eval/06_compute_final_summary.py
```

For human evaluation and final reporting:

```bash
python translation_eval/07_prepare_human_eval_package.py
python translation_eval/07_5_prepare_replacement_for_invalid_items.py
python translation_eval/08_compute_human_eval_summary.py
python translation_eval/09_make_final_report_tables.py
python translation_eval/10_make_final_result_section.py
```

---

## Repository Structure

```text
MLP_Emotion_Translation/
├─ README.md
├─ .gitignore
│
├─ modeling/
│  ├─ train_audio_only.py
│  ├─ train_multimodal.py
│  ├─ optimize_hyperparams.py
│  ├─ final_report_modeling.md
│  └─ README.md
│
└─ translation_eval/
   ├─ 00_inspect_audio_model_project.py
   ├─ 00_prepare_own_segments_from_srt.py
   ├─ 01_prepare_eval_inputs.py
   ├─ 02_predict_audio_ef_vae.py
   ├─ 03_generate_translation_audio_pred_deepseek.py
   ├─ 04_predict_va_vadbert.py
   ├─ 05_evaluate_quality_deepseek_blind.py
   ├─ 06_compute_final_summary.py
   ├─ 07_prepare_human_eval_package.py
   ├─ 07_5_prepare_replacement_for_invalid_items.py
   ├─ 08_human_eval_summary_simple.py
   ├─ 08_compute_human_eval_summary.py
   ├─ 09_make_final_report_tables.py
   ├─ 10_make_final_result_section.py
   ├─ 10_final_result_section_KR.md
   ├─ final_results_writeup.md
   ├─ human_eval_package/
   └─ human_eval_replacement_package/
```

---

## Evaluation Design

The evaluation compares two translation conditions:

| Condition         | Description                                                                          |
| ----------------- | ------------------------------------------------------------------------------------ |
| Baseline      | Korean text-only translation                                                         |
| Emotion-Aware | Translation using predicted emotion, valence, arousal, and EF-related affective cues |

The evaluation consists of three parts:

1. **Automatic V/A Preservation**
   Measures whether valence and arousal tendencies are preserved after translation.

2. **LLM-Based Blind Evaluation**
   Uses an LLM judge to compare baseline and emotion-aware translations on semantic fidelity, emotion consistency, fluency, and overall preference.

3. **Human Blind A/B Evaluation**
   Human annotators compare two anonymized translations without knowing which system produced them.

---

## Results Summary

### Human Blind A/B Evaluation

| Metric                  | Baseline Wins | Emotion-Aware Wins | Ties | Main Result                       |
| ----------------------- | ------------: | -----------------: | ---: | --------------------------------- |
| Semantic Fidelity       |            43 |                 36 |   71 | Baseline slightly favored         |
| **Emotion Consistency** |            20 |             **62** |   68 | **Emotion-aware clearly favored** |
| Fluency                 |            36 |                 36 |   78 | No clear difference               |
| Overall Preference      |            44 |                 64 |   42 | Emotion-aware moderately favored  |

### Human Score-Based Evaluation

| Metric                  | Mean Difference<br>(Emotion-Aware - Baseline) | Wilcoxon p-value | Interpretation              |
| ----------------------- | --------------------------------------------: | ---------------: | --------------------------- |
| Semantic Fidelity       |                                        -0.142 |         0.150677 | Not significant             |
| **Emotion Consistency** |                                    **+0.390** |     **0.000195** | **Significant improvement** |
| Fluency                 |                                        +0.033 |         0.826030 | Not significant             |

### LLM-Based Blind Evaluation

| Metric                  | Baseline Wins | Emotion-Aware Wins | Ties | Interpretation                     |
| ----------------------- | ------------: | -----------------: | ---: | ---------------------------------- |
| Semantic Fidelity       |            84 |                122 |  294 | Mostly tied                        |
| **Emotion Consistency** |            34 |            **227** |  239 | **Emotion-aware strongly favored** |
| Fluency                 |            85 |                 94 |  321 | Mostly tied                        |
| Overall Preference      |           104 |                238 |  158 | Emotion-aware favored              |

### Inter-Annotator Agreement

| Metric              | Complete Agreement Rate | Fleiss' Kappa |
| ------------------- | ----------------------: | ------------: |
| Semantic Fidelity   |                    0.36 |         0.164 |
| Emotion Consistency |                    0.40 |         0.228 |
| Fluency             |                    0.44 |         0.317 |
| Overall Preference  |                    0.56 |         0.467 |

---

## Main Finding

The proposed emotion-aware translation setting does **not** uniformly improve general translation quality.

Instead, its strongest contribution is in emotion consistency:

> Audio/EF-informed affective prompting significantly improves human-rated emotion consistency while leaving semantic fidelity and fluency largely unchanged.

Therefore, this project should be interpreted as an emotion-preservation translation framework, rather than a general-purpose translation-quality improvement method.

---

## Modeling Component

The `modeling/` folder contains the emotion recognition backbone of the project.

It includes:

* an audio-only baseline model,
* an audio + EF multimodal model,
* EF-based emotion weighting,
* KEMDy19 soft-label processing,
* and modeling reports.

The trained model produces the affective signals used in the downstream translation evaluation pipeline.

---

## Human Evaluation Note

Human evaluation was conducted as a blind A/B comparison.

The human evaluation files are anonymized.
A/B condition mapping files are included only after annotation completion to support reproducibility.

If this repository is made public beyond the project context, users should review whether the human evaluation answer-key files should remain included.

---

## Privacy and Release Notes

This repository should not include:

* raw audio files,
* API keys,
* local environment files,
* unlicensed private data,
* or unnecessary large intermediate outputs.

Model checkpoints may be included for reproducibility, but they can also be replaced with external download links if repository size becomes an issue.

---
## Contributors

This repository contains two connected components of the Emotion-Aware Korean Translation Program.

| Component                    | Main Contributor | Description                                                                                                                                                                                         
| Emotion recognition modeling | Kim Tae-hyun     | Built the KEMDy19-based audio/EF emotion recognition model and prepared the modeling pipeline.                                                                                                                                                    |
| Translation evaluation       | Li Xiaomin       | Designed and implemented the Korean-to-English emotion-aware translation evaluation pipeline, including automatic V/A evaluation, LLM-based blind evaluation, human A/B evaluation, replacement annotation processing, and final result analysis. |

The modeling component provides the affective prediction signals used by the downstream translation evaluation pipeline.

## Contact

For questions, please open an issue in this repository.