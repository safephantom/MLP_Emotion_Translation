# Translation Evaluation Pipeline (translation_eval)

**English Version** | 🔗 **[한국어 버전 (Korean Version)](./README.ko.md)**

---

## 1. Overview & Evaluation Paradigm
This directory implements the downstream **translation generation and multi-dimensional evaluation pipeline**. 

The goal of this evaluation framework is to verify whether injecting predicted acoustic and linguistic emotional signals (Valence, Arousal, Emotion Category) into a Large Language Model (LLM) translation system improves translation quality, specifically by preserving the speaker's emotional stance without losing semantic information.

The evaluation compares two conditions:
1. **Baseline**: Text-only translation prompt.
2. **Emotion-Aware (Proposed)**: Translation prompt enriched with Valence, Arousal, and Emotion categories predicted by the front-end multimodal LSTM model.

---

## 2. Step-by-Step Pipeline Execution
The pipeline is structured into 11 logical steps, each mapping to a specific Python script:

### Step 0: Segment Preparation & Audio Inspection
* **`00_prepare_own_segments_from_srt.py`**: Extracts segment-level Korean texts and corresponding audio clips from source video/subtitles.
* **`00_inspect_audio_model_project.py`**: Inspects local configurations and model checkpoints to ensure compatibility.

### Step 1: Input Preprocessing
* **`01_prepare_eval_inputs.py`**: Standardizes the Korean source inputs and extracts linguistic features.

### Step 2: Front-End Affect Inference
* **`02_predict_audio_ef_vae.py`**: Uses the trained multimodal emotion model (Audio + EF) to predict:
  * Discrete Emotion Category
  * Valence (1 to 5 scale)
  * Arousal (1 to 5 scale)

### Step 3: LLM Translation Generation
* **`03_generate_translation_audio_pred_deepseek.py`**: Generates translations via DeepSeek-V3 API under the two experimental conditions.

### Step 4: Automated Emotion Estimation
* **`04_predict_va_vadbert.py`**: Estimates Valence and Arousal scores from the generated English translations using a pre-trained VAD-BERT regressor to measure emotion preservation automatically.

### Step 5: LLM-Based Blind Quality Evaluation
* **`05_evaluate_quality_deepseek_blind.py`**: Executes an LLM-based blind A/B judgment on translation pairs across semantic fidelity, emotion consistency, fluency, and overall preference.

### Step 6: Compilation of Automatic Metrics
* **`06_compute_final_summary.py`**: Summarizes the automatic scores, including statistical correlations between source V/A and target V/A.

### Step 7 & 7.5: Human Evaluation Sheet Generation
* **`07_prepare_human_eval_package.py`**: Packages translations into anonymized sheets for human evaluation.
* **`07_5_prepare_replacement_for_invalid_items.py`**: Identifies outliers or invalid ratings in human feedback and generates replacement validation sheets.

### Step 8 & 8.5: Human Evaluation Compilation
* **`08_compute_human_eval_summary.py`**: Compiles raw human annotations, calculates mean difference scores, and conducts Wilcoxon Signed-Rank tests for significance.
* **`08_human_eval_summary_simple.py`**: Outputs simplified summary metrics.

### Step 9 & 10: Reporting & Writing
* **`09_make_final_report_tables.py`**: Automates table formatting for LaTeX and Markdown.
* **`10_make_final_result_section.py`**: Integrates quantitative and qualitative outputs into a cohesive evaluation report.

---

## 3. Experimental Evaluation Results

Our evaluation setup involves:
1. **Automated V/A Correlation**: Quantifying preservation of emotional dimensions.
2. **LLM-Based Blind Evaluation**: $N=500$ evaluation pairs.
3. **Human Blind A/B Evaluation**: Triple-annotated blind trials.

### 3.1. Human Blind A/B Evaluation ($N = 150$)

| Evaluation Dimension | Baseline Wins | Proposed Wins | Ties | Overall Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| Semantic Fidelity | 43 | 36 | 71 | Baseline slightly favored (No stat. diff) |
| **Emotion Consistency** | 20 | **62** | 68 | **Proposed clearly favored (Significant)** |
| Fluency | 36 | 36 | 78 | Identical performance |
| Overall Preference | 44 | **64** | 42 | Proposed moderately favored |

### 3.2. Human Score-Based Statistical Analysis (Wilcoxon Signed-Rank Test)

| Metric | Mean Score Diff. (Proposed - Baseline) | Wilcoxon $p$-value | Statistical Significance |
| :--- | :---: | :---: | :--- |
| Semantic Fidelity | -0.142 | 0.1507 | Not Significant ($p \ge 0.05$) |
| **Emotion Consistency** | **+0.390** | **0.000195** | **Highly Significant ($p < 0.001$)** |
| Fluency | +0.033 | 0.8260 | Not Significant ($p \ge 0.05$) |

### 3.3. LLM-Based Blind Evaluation ($N = 500$)

| Evaluation Dimension | Baseline Wins | Proposed Wins | Ties | Overall Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| Semantic Fidelity | 84 | 122 | 294 | Comparable |
| **Emotion Consistency** | 34 | **227** | 239 | **Proposed strongly favored** |
| Fluency | 85 | 94 | 321 | Comparable |
| Overall Preference | 104 | **238** | 158 | Proposed favored |

### 3.4. Inter-Annotator Agreement (Human Evaluators)

| Metric | Complete Agreement Rate | Fleiss' Kappa | Agreement Strength |
| :--- | :---: | :---: | :--- |
| Semantic Fidelity | 36% | 0.164 | Slight Agreement |
| Emotion Consistency | 40% | 0.228 | Fair Agreement |
| Fluency | 44% | 0.317 | Fair Agreement |
| **Overall Preference** | **56%** | **0.467** | **Moderate Agreement** |

---

## 4. Key Insights & Conclusion
1. **Emotion Preservation vs. Translation Quality**: Injecting Valence/Arousal/Emotion cues does not lead to a uniform upgrade in raw language translation (fluency or general lexicon). However, it shows a **clear, statistically validated improvement in preserving speaker stance and emotional consistency** ($p < 0.001$).
2. **Robustness of the LLM Prompting**: The DeepSeek model successfully utilizes the injected VAE context to alter lexical choices (e.g., changing endings, utilizing emotion-specific vocabulary) to match the target tone without hallucinating or damaging the underlying semantic meaning.