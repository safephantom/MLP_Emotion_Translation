# KEMDy19-based Multimodal Emotion Recognition (Modeling Backbone)

**English Version** | 🔗 **[한국어 버전](./README.md)**

---

This directory contains the front-end **emotion recognition backbone** of the project. It leverages the KEMDy19 dataset to train a multimodal neural network that predicts discrete emotion classes alongside dimensional affective metrics—**Valence and Arousal (V/A)**. 

The core novelty lies in combining **acoustic features** from raw audio with **sentence-final endings (Ending Forms, EF)** which act as crucial structural cues in spoken Korean emotion expression.

---

## 1. Methodology & Modeling Strategy

### 1.1. Soft Labeling (Label Distribution Learning)
Unlike traditional classifiers that enforce a single ground truth (Hard Label / One-hot Encoding), this model utilizes **Soft Labeling**. For each utterance in KEMDy19, 10 human evaluators voted on the emotional category. We computed the vote distribution (e.g., 60% Anger, 40% Sadness) and trained the network to match this probability distribution. This approach allows the network to learn rich, blended emotional nuances.

### 1.2. Multi-task Learning
To gain a holistic representation of emotional states, the model is designed to simultaneously predict:
1. **Discrete Emotion Classification (E)**: Soft probability distribution across emotion classes (Anger, Sadness, Happiness, Fear, Disgust, Surprise, Neutral).
2. **Valence (V) Regression**: Valence measures how positive or negative the speaker's emotional state is.
3. **Arousal (A) Regression**: Arousal measures the physical intensity of the emotion.

The joint loss function combines Cross-Entropy (for emotion classes) and Mean Squared Error (for Valence and Arousal):
$$
\mathcal{L}_{total} = \alpha \cdot \mathcal{L}_{Emotion} + \beta \cdot \mathcal{L}_{Valence} + \gamma \cdot \mathcal{L}_{Arousal}
$$

### 1.3. Hyperparameter Optimization via Optuna
To establish a strong performance ceiling, we integrated the Optuna framework. Over 50 search trials, we identified the optimal network hyperparameters:
* **Learning Rate**: 0.0039
* **Batch Size**: 32
* **LSTM Hidden Dimension**: 128
* **Dropout Rate**: 0.41

---

## 2. Hypothesis Verification: Ablation Study
We conducted an **Ablation Study** to verify the core hypothesis: *Korean sentence-final endings (EF) are crucial in conveying emotional nuance.*
* **Proposed (Multi-modal)**: Audio + Text (EF probability distribution).
* **Ablation (Audio-Only)**: Audio with EFs masked via uniform distribution ($1/7$ for each class).

### Experimental Results

| Model Setup | Modality | Best Validation Loss | Hard Accuracy | Soft Accuracy | V Loss (MSE) | A Loss (MSE) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Multi-modal (Proposed)** | **Audio + Text(EF)** | **1.4470** | **59.17%** | **54.44%** | **0.0501** | **0.0168** |
| **Audio-Only (Ablated)** | Audio Only (EF Masked) | 1.6808 | 44.28% | 46.14% | 0.0660 | 0.0203 |

#### Discussion
* **Drop in Accuracy**: Masking EF information caused the Hard Accuracy to drop significantly by **~15%p** (from 59.17% to 44.28%). This demonstrates that acoustic cues alone are insufficient to resolve the final emotion in spoken Korean; EFs act as the linguistic punctuation that resolves emotional ambiguity.
* **Valence Error Elevation**: The Valence loss increased by ~32%, indicating that EFs are strong indicators of whether an utterance leans positive or negative.

---

## 3. Directory Structure & Files

* **`02_train_multimodal.py`**: The training script for the multimodal network combining audio features and EF weights.
* **`03_train_audio_only.py`**: The ablated baseline script where the EF features are masked using a uniform distribution.
* **`01_optimize_hyperparams.py`**: Optuna search script for hyperparameter optimization.
* **`final_report_modeling.md`**: Detailed research and technical report covering modeling and tuning decisions.
* **`merged_dataset_soft_fixed.csv`**: Cleansed KEMDy19 labels with soft probability scores.
* **`dynamic_ef_weights_fixed.csv`**: Pre-calculated probability maps of emotions given specific ending forms (EF).
* **`kemdy19_multimodal_lstm.pth`**: Trained weight parameters of the best multimodal model.
* **`kemdy19_audio_only.pth`**: Trained weight parameters of the ablated audio-only model.

---

## 4. Usage & Replication
To retrain or run evaluations:
1. Ensure the audio features are cached (or download the pre-extracted `cached_features/` zip file).
2. Execute the training script:
   ```bash
   python modeling/02_train_multimodal.py
   ```
3. To search for new hyperparameters:
   ```bash
   python modeling/01_optimize_hyperparams.py
   ```
