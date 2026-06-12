## MLP_Emotion_Translation

2026-1 <기계학습과 딥러닝> 수업의 기말 프로젝트 결과물입니다.

# README.md

````markdown
# Emotion-Aware Korean Translation Program

This project proposes an **emotion-aware Korean translation framework** that incorporates **acoustic emotion cues** and **Korean sentence-final endings (EF, ending forms)** into the translation pipeline.  
The goal is to move beyond semantic-only translation and improve the preservation of **emotional nuance, speaker attitude, and interactional context** in Korean-to-English translation.

## 1. Motivation and Background

Conventional translation systems often focus mainly on propositional meaning and surface lexical content.  
However, in Korean spoken language, a speaker’s intended meaning is shaped not only by lexical semantics, but also by:

- acoustic features (e.g., prosody, emotional intensity),
- sentence-final endings (EF), which strongly reflect attitude and stance,
- and the interaction between emotional tone and utterance form.

This project is motivated by the need for a translation framework that preserves not only *what is said*, but also *how it is said*.

## 2. Goal of the Project

The project aims to build and evaluate an emotion-aware Korean translation program that:

1. analyzes Korean speech and expressive linguistic features,
2. predicts emotion-related signals such as **emotion / valence / arousal**,
3. injects those signals into the translation stage,
4. and compares the resulting translations against a baseline translation system.

More specifically, the project investigates whether a **speech-feature + EF-assisted model** can:

- improve translation quality, and
- preserve the emotional characteristics of the source utterance.

## 3. Core Idea

The overall idea of the system is:

```text
Korean speech + Korean text + EF information
        ↓
Emotion prediction model
        ↓
Predicted emotion / valence / arousal
        ↓
Emotion-aware translation prompting
        ↓
English translation
        ↓
Automatic + human evaluation
````

The project assumes that **audio-derived affective information** and **Korean sentence-final endings** provide useful cues for more emotionally faithful translation.

## 4. Dataset and Modeling Background

The core emotion modeling component is based on KEMDy19, a Korean multimodal emotion dataset.
Using this dataset, an emotion prediction model was trained to capture:

* emotion category
* valence
* arousal

The modeling framework also incorporates **Korean sentence-final endings (EF)** as an auxiliary cue, based on the hypothesis that EF plays a critical role in Korean emotional expression.

An ablation study showed that combining **audio features + EF information** outperforms an audio-only setting in emotion recognition, supporting the use of this model as a front-end signal provider for translation.

## 5. Translation Evaluation Objective

The evaluation part of this repository focuses on verifying whether the acoustic-feature + EF-assisted translation setting can:

1. improve translation quality, and
2. preserve the emotional properties of the source utterance.

To make the evaluation feasible within a short time window, the project adopts a fast and reproducible evaluation framework based on:

* automatic V/A preservation analysis,
* LLM-based blind quality evaluation,
* and human A/B evaluation.

## 6. Overall Evaluation Framework

The evaluation consists of **two independent dimensions**.

### 6.1 Emotion Preservation

Emotion preservation is measured by comparing the source-side and translation-side values of:

* valence
* arousal

The source-side valence/arousal signals come from the Korean-side prediction pipeline, and the translation-side values are estimated from the English translation output.
This allows us to quantify whether the translated sentence preserves the emotional tendency of the original utterance.

### 6.2 Translation Quality

Translation quality is evaluated through LLM-based blind judgment and human annotation.
The main evaluation dimensions are:

* semantic fidelity
* emotion consistency
* fluency
* overall preference

The final comparison is made between:

* Baseline model: text-only translation
* Target model: acoustic + EF-informed emotion-aware translation

## 7. What This Repository Contains

This repository mainly contains the translation evaluation pipeline built on top of the broader Emotion-Aware Korean Translation Program.

In other words:

* the main project is the emotion-aware Korean translation framework,
* and this repository documents the evaluation workflow used to test whether the framework actually helps.

## 8. Repository Structure

```text
MLP_Emotion_Translation/
├─ README.md
├─ .gitignore
│
├─ modeling/
│  ├─ README.md
│  ├─ train_audio_only.py
│  ├─ train_multimodal.py
│  ├─ optimize_hyperparams.py
│  ├─ dynamic_ef_weights_fixed.csv
│  ├─ merged_dataset_soft_fixed.csv
│  ├─ kemdy19_audio_only.pth
│  └─ kemdy19_multimodal_lstm.pth
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

## 9. Translation Evaluation Pipeline

The translation evaluation pipeline proceeds as follows.

### Step 0. Segment preparation

Prepare segment-level Korean text and corresponding audio clips from source audio and subtitle files.

```bash
python translation_eval/00_prepare_own_segments_from_srt.py
```

### Step 1. Evaluation input preparation

Clean and organize Korean source text, and extract EF-related information where needed.

```bash
python translation_eval/01_prepare_eval_inputs.py
```

### Step 2. Audio-based affect prediction

Use the trained emotion model to predict:

* emotion
* valence
* arousal

from audio (with EF-related support).

```bash
python translation_eval/02_predict_audio_ef_vae.py
```

### Step 3. Translation generation

Generate two translation conditions:

* baseline translation
* audio-prediction-based emotion-aware translation

```bash
python translation_eval/03_generate_translation_audio_pred_deepseek.py
```

### Step 4. V/A preservation evaluation

Predict valence and arousal from the translated English outputs and compute preservation metrics.

```bash
python translation_eval/04_predict_va_vadbert.py
```

### Step 5. LLM-based blind evaluation

Conduct blind A/B comparison using an LLM judge over:

* semantic fidelity
* emotion consistency
* fluency
* overall preference

```bash
python translation_eval/05_evaluate_quality_deepseek_blind.py
```

### Step 6. Automatic summary

Summarize automatic evaluation results and condition-wise comparisons.

```bash
python translation_eval/06_compute_final_summary.py
```

### Step 7. Human evaluation package generation

Generate human evaluation sheets for annotators.

```bash
python translation_eval/07_prepare_human_eval_package.py
```

### Step 7.5. Replacement annotation generation

Identify invalid items from human annotation and prepare replacement items.

```bash
python translation_eval/07_5_prepare_replacement_for_invalid_items.py
```

### Step 8. Human evaluation summary

Summarize human annotation results and inter-annotator agreement.

```bash
python translation_eval/08_compute_human_eval_summary.py
```

### Step 9. Final report table generation

Generate integrated result tables for reporting.

```bash
python translation_eval/09_make_final_report_tables.py
```

### Step 10. Final interpretation writing

Generate the final Korean write-up section for result interpretation.

```bash
python translation_eval/10_make_final_result_section.py
```

## 10. Main Evaluation Logic

The evaluation compares two conditions:

### Baseline

Korean text
→ English translation

### Target

Korean speech + EF-informed affect prediction
→ predicted emotion / valence / arousal
→ Korean text + predicted affective signals
→ English translation

This design allows the project to test whether affective prompting improves translation beyond a standard text-only baseline.

## 11. Main Findings

The current results suggest that the audio-model emotion-aware condition does **not uniformly improve all aspects of translation quality**.
Instead, its main contribution lies in **emotion consistency** and the preservation of speaker attitude.

In summary:

* **emotion consistency** tends to improve most clearly,
* **semantic fidelity** does not show a clear overall gain,
* **fluency** remains largely similar,
* and **automatic V/A preservation** shows only a weak positive tendency.

Therefore, the proposed approach is best interpreted as a method for emotion-aware translation enhancement, rather than a universal quality-improvement method.

## 12. Human Evaluation Note

Human evaluation in this project was conducted as a **blind A/B comparison**.

Annotators compared:

* translation A
* translation B

without knowing which condition was baseline and which was emotion-aware.
A/B condition mapping was restored only after annotation was completed.

The human evaluation files in this repository are anonymized and included for reproducibility.

## 13. Environment Setup

Install the main dependencies:

```bash
pip install pandas numpy scipy openpyxl torch transformers librosa soundfile tqdm openai
```

If DeepSeek API is used, set the API key before running relevant scripts:

```powershell
$env:DEEPSEEK_API_KEY="your_api_key_here"
```

## 14. Notes

* Raw audio files, API keys, and local environment files should not be uploaded.
* Some outputs depend on local paths and external API/model access.
* Some evaluation files are regenerated by the later scripts and may not need to be stored permanently.

## 15. Project Summary

Emotion-Aware Korean Translation Program is a project that integrates:

* Korean acoustic emotion information,
* Korean sentence-final endings (EF),
* and translation evaluation

to investigate whether emotionally enriched signals can improve Korean-to-English translation.

The main contribution of the project is not simply better translation in a general sense, but more faithful preservation of:

* emotional nuance,
* speaker stance,
* and interactional meaning.


---
## Contributors

This repository contains two connected components of the Emotion-Aware Korean Translation Program.

| Component                    | Main Contributor | Description                                                                                                                                                                                         
| Emotion recognition modeling | Kim Tae-hyun     | Built the KEMDy19-based audio/EF emotion recognition model and prepared the modeling pipeline.                                                                                                                                                    |
| Translation evaluation       | Li Xiaomin       | Designed and implemented the Korean-to-English emotion-aware translation evaluation pipeline, including automatic V/A evaluation, LLM-based blind evaluation, human A/B evaluation, replacement annotation processing, and final result analysis. |

The modeling component provides the affective prediction signals used by the downstream translation evaluation pipeline.

## Contact

For questions, please open an issue in this repository.