# Emotion-Aware Korean Translation Program (Emotion-Translation)
> **Graduate Course Project for Machine Learning and Programming (2026-1)**

🔗 **[한국어 버전 (Korean Version)](./README.ko.md)** | **English Version**

---

## 1. Abstract & Motivation
Conventional machine translation frameworks predominantly focus on preserving propositional meanings and lexical semantics. However, in spoken Korean, emotional nuances, speaker attitudes, and interactional stances are significantly shaped by:
1. **Acoustic Features**: Pitch, intensity, and prosodic variations.
2. **Sentence-Final Endings (Ending Forms, EF)**: Grammaticalized components that structurally express speech styles, politeness, and speaker stance.

This project proposes an **emotion-aware Korean-to-English translation framework** that leverages a **multimodal emotion recognition model** as a front-end signal provider. This model predicts discrete emotions along with dimensional affective values—**Valence and Arousal (V/A)**—and injects these signals into a Large Language Model (LLM) translation pipeline. 

---

## 2. System Architecture & Core Pipeline
Our framework processes spoken Korean input and generates an emotion-preserved English translation through the following pipeline:

```text
Korean Speech Audio + Korean Source Text + Ending Forms (EF)
                            │
                            ▼
           [ Multimodal Emotion Predictor ] (LSTM)
         - Predicts: Emotion Class, Valence, Arousal
                            │
                            ▼
        [ Emotion-Aware Translation Prompting ] (DeepSeek)
                            │
                            ▼
          [ Emotion-Preserving English Translation ]
                            │
                            ▼
        [ Multi-Dimensional Translation Evaluation ]
         - V/A Preservation Analysis (VAD-BERT)
         - LLM-based Blind Quality Evaluation
         - Human Blind A/B Evaluation
```

---

## 3. Directory Structure
The repository is structured into two main logical components:

```text
MLP_Emotion_Translation/
├─ README.md                      # English Main Documentation
├─ README.ko.md                   # Korean Main Documentation
├─ .gitignore
│
├─ modeling/                      # Emotion Recognition Model Backbone
│  ├─ README.md                   # Modeling Documentation (Korean)
│  ├─ README.en.md                # Modeling Documentation (English)
│  ├─ 02_train_multimodal.py      # Multi-modal training (Audio + EF)
│  ├─ 03_train_audio_only.py      # Baseline training (Ablation setup)
│  ├─ 01_optimize_hyperparams.py  # Hyperparameter search via Optuna
│  ├─ final_report_modeling.md    # In-depth modeling report
│  ├─ merged_dataset_soft_fixed.csv # Emotion label distributions (Soft Labels) from 10 annotators
│  └─ dynamic_ef_weights_fixed.csv  # Pre-calculated probability weights mapping Korean EFs to emotions
│
└─ translation_eval/              # Translation Generation & Evaluation
   ├─ README.md                   # Evaluation Documentation (English)
   ├─ README.ko.md                # Evaluation Documentation (Korean)
   ├─ 01_prepare_eval_inputs.py   # Prep input text and extract EFs
   ├─ 02_predict_audio_ef_vae.py  # Infer Emotion & V/A labels
   ├─ 03_generate_translation_audio_pred_deepseek.py  # Translation generation
   ├─ 04_predict_va_vadbert.py    # Predict V/A from translations
   ├─ 05_evaluate_quality_deepseek_blind.py  # LLM blind A/B judgment
   ├─ 06_compute_final_summary.py # Compile automatic eval metrics
   ├─ 07_prepare_human_eval_package.py  # Prepare human evaluation sheets
   ├─ 08_compute_human_eval_summary.py  # Compile human evaluation results
   ├─ 09_make_final_report_tables.py    # Generate tables for report
   └─ 10_make_final_result_section.py   # Create final interpretation md
```

* For details regarding the front-end emotion prediction models, please refer to the [modeling/README.md](file:///c:/Project/MLP_Emotion_Translation/modeling/README.md).
* For detailed steps of the downstream translation and evaluation pipeline, please refer to the [translation_eval/README.md](file:///c:/Project/MLP_Emotion_Translation/translation_eval/README.md).

---

## 4. Key Experimental Findings

### 4.1. Front-End Emotion Classifier (Ablation Study)
Before feeding predictions into the translation pipeline, we validated the multimodal LSTM emotion classifier. The ablation study proved that incorporating Korean sentence-final endings (EF) alongside raw audio features dramatically reduces emotional ambiguity:

| Model Setup | Modality | Hard Accuracy | Valence Loss (MSE) | Arousal Loss (MSE) |
| :--- | :--- | :---: | :---: | :---: |
| **Multimodal (Proposed)** | **Audio + EF** | **59.17%** | **0.0501** | **0.0168** |
| **Audio-Only (Ablated)** | Audio Only (EF Masked) | 44.28% | 0.0660 | 0.0203 |

* **Key Insight**: Excluding grammatical EF weights causes classification accuracy to drop by **~15%p**, validating that EFs act as primary contextual punctuation for emotional resolution in spoken Korean.
  *(Note: Valence and Arousal loss metrics are evaluated based on Mean Squared Error (MSE).)*

### 4.2. Downstream Translation Performance
Our downstream translation evaluations compared the baseline (text-only translation) against the proposed model (audio + EF-informed emotion-aware translation).

Through automated metrics, LLM-based blind judges, and human annotation, we found that:
* **Emotion Consistency (Statistically Significant)**: Incorporating predicted emotion values (VAE) significantly improves the emotional fidelity of translations (Wilcoxon Signed-Rank Test: $p < 0.001$).
* **Semantic Fidelity & Fluency (Comparable)**: The injection of emotional prompt signals does not compromise semantic correctness or translation naturalness, resulting in statistically comparable performance to the baseline.

| Evaluation Method | Metric | Baseline Wins | Proposed (Emotion-Aware) Wins | Ties |
| :--- | :--- | :---: | :---: | :---: |
| **Human Blind A/B** | Emotion Consistency | 20 | **62** | 68 |
| **LLM Blind A/B** | Emotion Consistency | 34 | **227** | 239 |

> 📌 **Academic Conclusion**: The proposed method functions as an effective **emotion-preservation layer** that enriches translations with speaker nuances without degrading semantic content.

---

## 5. Quick Start & Setup

### Environment Setup
Install the necessary python dependencies:
```bash
pip install pandas numpy scipy openpyxl torch transformers librosa soundfile tqdm openai
```

Set up your DeepSeek API key (used for LLM translation and quality evaluations):
```bash
# bash
export DEEPSEEK_API_KEY="your_api_key_here"

# PowerShell
$env:DEEPSEEK_API_KEY="your_api_key_here"
```

### Execution Steps

#### Component A: Emotion Recognition Modeling (Training Phase)
1. Run hyperparameter optimization to find best configurations (Optional):
   ```bash
   python modeling/01_optimize_hyperparams.py
   ```
2. Train the multimodal model (Audio + EF) or audio-only baseline:
   ```bash
   python modeling/02_train_multimodal.py
   python modeling/03_train_audio_only.py
   ```

#### Component B: Translation & Evaluation Pipeline
1. Run the front-end features and affect predictions:
   ```bash
   python translation_eval/01_prepare_eval_inputs.py
   python translation_eval/02_predict_audio_ef_vae.py
   ```
2. Generate translations and predict emotional dimensions:
   ```bash
   python translation_eval/03_generate_translation_audio_pred_deepseek.py
   python translation_eval/04_predict_va_vadbert.py
   ```
3. Run evaluation scripts (LLM blind evaluation and human evaluation summarization):
   ```bash
   python translation_eval/05_evaluate_quality_deepseek_blind.py
   python translation_eval/06_compute_final_summary.py
   ```

For a comprehensive walkthrough of the evaluation pipeline, see the [translation_eval/README.md](file:///c:/Project/MLP_Emotion_Translation/translation_eval/README.md).

## Qualitative Translation Examples

While quantitative evaluations demonstrate improvements in emotion consistency, qualitative examples provide a clearer view of how emotional information influences translation outputs.

The following examples compare translations generated by:

* **Baseline Translation** (without emotion information)
* **Emotion-aware Translation** (with predicted emotion information)

### 1. Punctuation Enhancement

Emotion-aware translation tends to preserve emotional intensity through punctuation marks such as exclamation marks and question marks, resulting in expressions that are closer to the speaker's original emotional state.

**Source (Korean)**

[이게 맥시멈이]

**Baseline Translation**

[This is the maximum.]

**Emotion-aware Translation**

[This is the maximum?]

**Observation**

The emotion-aware translation better reflects the speaker's emotional intensity through punctuation usage, making the translated utterance more expressive.

### 2. Tone and Attitude Adjustment

Beyond literal meaning, emotional information can influence the overall tone of an utterance. Emotion-aware translation often selects wording that more accurately conveys the speaker's attitude, such as frustration, excitement, disappointment,or surprise.

**Source (Korean)**

[뭐야]

**Baseline Translation**

[What is it]

**Emotion-aware Translation**

[What the...?]

**Observation**

The emotion-aware translation captures the speaker's intended attitude more naturally, resulting in a translation that better aligns with the emotional context.

### 3. Korean Sentence-Final Ending Preservation

Korean sentence-final endings often carry important emotional and interpersonal information. By incorporating emotion predictions, the translation system is better able to preserve the pragmatic effect of these endings in the target language.

**Source (Korean)**

[뭐 그런 분들도 계실 텐데]

**Baseline Translation**

[Well, there might be people like that.]

**Emotion-aware Translation**

[Well, there might be people like that too, I suppose.]

**Observation**

The emotion-aware translation more effectively conveys the emotional nuance embedded in the Korean sentence-final ending, producing a translation that feels closer to the original speaker's intent.

### Summary

Across these examples, emotion-aware translation demonstrates improvements in:

* Emotional punctuation preservation
* Tone and attitude consistency
* Sentence-final ending interpretation

These qualitative observations are consistent with the quantitative evaluation results, which show substantial gains in emotion consistency while maintaining comparable semantic accuracy.
