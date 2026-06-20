# 감정 인지형 한국어-영어 번역 프로그램 (Emotion-Translation)
> **기계학습과 프로그래밍 대학원 과정 기말 프로젝트 (2026-1)**

English Version: 🔗 **[English Version](./README.md)** | **한국어 버전**

---

## 1. 초록 및 연구 동기
기존의 기계 번역 프레임워크는 주로 문맥적 명제 의미와 어휘적 의미론의 보존에 초점을 맞추어 왔습니다. 그러나 한국어 구어체 발화에서는 화자의 감정적 뉘앙스, 태도, 대화적 스탠스가 다음 두 가지 요소에 의해 크게 좌우됩니다.
1. **음향 자질**: 피치, 강도 및 운율적 변화.
2. **문장 종결어미(EF)**: 상대 높임법, 문체 및 화자의 태도를 문법적으로 표현하는 한국어의 핵심 구조적 요소.

본 프로젝트는 **멀티모달 감정 인식 모델**을 전처리 엔진으로 활용하는 **감정 인지형 한-영 번역 프레임워크**를 제안합니다. 전처리 모델은 음향 데이터와 종결어미 분포를 결합하여 이산 감정 범주 및 차원적 감정값인 **Valence(긍정/부정) 및 Arousal(활성도)**을 예측합니다. 예측된 정서적 시그널을 거대 언어 모델(LLM, DeepSeek) 기반 번역 프롬프트에 주입함으로써, 문장의 의미론적 정확성을 훼손하지 않으면서 화자의 **감정적 충실도**를 함께 보존하는 번역을 수행합니다.

---

## 2. 시스템 아키텍처 및 핵심 파이프라인
본 프레임워크는 입력된 한국어 음성 및 텍스트 데이터를 처리하여 감정이 보존된 영문 번역을 아래 파이프라인에 따라 생성합니다.

```text
한국어 발화 음성 + 한국어 소스 텍스트 + 종결어미(EF) 정보
                            │
                            ▼
           [ 멀티모달 감정 예측기 ] (LSTM)
         - 예측값: 감정 범주, Valence, Arousal
                            │
                            ▼
        [ 감정 인지형 번역 프롬프팅 ] (DeepSeek)
                            │
                            ▼
          [ 감정이 보존된 영문 번역 결과물 ]
                            │
                            ▼
            [ 다차원 번역 평가 파이프라인 ]
          - V/A 정서 보존도 분석 (VAD-BERT)
          - LLM 기반 블라인드 품질 평가 (LLM Judge)
          - 인간 평가자 기반 블라인드 A/B 테스트
```

---

## 3. 디렉토리 구조
본 리포지토리는 크게 두 개의 핵심 컴포넌트로 구성되어 있습니다.

```text
MLP_Emotion_Translation/
├─ README.md                      # 영문 메인 설명서
├─ README.ko.md                   # 국문 메인 설명서
├─ .gitignore
│
├─ modeling/                      # 감정 인식 백본 모델링 파트
│  ├─ README.md                   # 모델링 설명서 (한국어)
│  ├─ README.en.md                # 모델링 설명서 (영어)
│  ├─ 02_train_multimodal.py      # 멀티모달(음성 + 종결어미) 학습 스크립트
│  ├─ 03_train_audio_only.py      # 음향 단독 baseline 학습 스크립트 (Ablation)
│  ├─ 01_optimize_hyperparams.py  # Optuna 기반 하이퍼파라미터 탐색 스크립트
│  ├─ final_report_modeling.md    # 상세 모델링 연구 보고서
│  ├─ merged_dataset_soft_fixed.csv # 10명의 평가자 감정 투표 분포(Soft Label)가 반영된 데이터셋
│  └─ dynamic_ef_weights_fixed.csv  # 한국어 종결어미별 감정 카테고리 매핑 확률 테이블
│
└─ translation_eval/              # 번역 생성 및 다차원 평가 파트
   ├─ README.md                   # 번역 평가 설명서 (영어)
   ├─ README.ko.md                # 번역 평가 설명서 (한국어)
   ├─ 01_prepare_eval_inputs.py   # 텍스트 정제 및 종결어미 추출
   ├─ 02_predict_audio_ef_vae.py  # 학습된 모델을 통한 감정 및 V/A 추론
   ├─ 03_generate_translation_audio_pred_deepseek.py  # LLM 기반 번역 생성
   ├─ 04_predict_va_vadbert.py    # 번역된 영문 텍스트로부터 V/A 추론
   ├─ 05_evaluate_quality_deepseek_blind.py  # LLM 판정관 기반 A/B 블라인드 평가
   ├─ 06_compute_final_summary.py # 자동 평가 지표 취합 및 분석
   ├─ 07_prepare_human_eval_package.py  # 인간 평가용 테스트 세트 생성
   ├─ 08_compute_human_eval_summary.py  # 인간 평가 결과 분석 및 취합
   ├─ 09_make_final_report_tables.py    # 결과 테이블 생성 자동화
   └─ 10_make_final_result_section.py   # 정성/정량 분석 결과 마크다운 생성
```

* 감정 예측 전처리 모델 학습 및 구조에 대한 상세 정보는 [modeling/README.md](file:///c:/Project/MLP_Emotion_Translation/modeling/README.md)를 참고해 주세요.
* 감정 인지형 번역 생성 및 정량/정성 평가 실험 절차는 [translation_eval/README.md](file:///c:/Project/MLP_Emotion_Translation/translation_eval/README.md)를 참고해 주세요.

---

## 4. 주요 실험 결과 요약

### 4.1. 전면 감정 예측 모델 성능 (절제 연구)
번역 파이프라인에 모델 예측값을 주입하기에 앞서, 멀티모달 LSTM 감정 인식 모델의 검증을 수행했습니다. 절제 연구를 통해 한국어 음성 정보에 문장 종결어미(EF) 특징을 결합하는 것이 감정 판별의 모호성을 크게 낮추어 줌을 입증했습니다.

| 모델 설정 | 학습 모달리티 | Hard Accuracy (정확도) | Valence 손실 (MSE) | Arousal 손실 (MSE) |
| :--- | :--- | :---: | :---: | :---: |
| **멀티모달 (실험군)** | **음향 + 종결어미(EF)** | **59.17%** | **0.0501** | **0.0168** |
| **음향 단독 (대조군)** | 음향 단독 (EF 마스킹) | 44.28% | 0.0660 | 0.0203 |

* **핵심 인사이트**: 문법적 종결어미 가중치를 제외하는 경우 감정 분류 정확도가 **약 15%p** 하락하며, 종결어미가 한국어 구어체의 정서 파악에 있어 핵심적인 맥락 해결 장치임을 보여줍니다.
  *(참고: Valence와 Arousal 손실 평가는 평균제곱오차(MSE) 기준으로 측정되었습니다.)*

### 4.2. 감정 인지형 번역 성능 평가
번역 평가 실험은 대조군(기본 텍스트 번역)과 실험군(음향 및 종결어미 정보가 주입된 번역) 조건을 비교하여 진행되었습니다.

자동 정서 분석기(VAD-BERT), LLM Judge, 그리고 인간 평가단의 블라인드 A/B 테스트 결과는 다음과 같습니다.
* **감정 보존 일치도**: 제안된 감정 정보 주입 번역 방식이 번역문 내 화자의 감정선 보존력을 유의미하게 향상시켰습니다 (Wilcoxon 부호 순위 검정: $p < 0.001$).
* **의미 전달력 및 유창성**: 정서적 정보 주입 프롬프트가 추가되더라도 의미론적 정확도와 문법적 유창성은 기존 텍스트 번역과 동등한 수준을 유지하였습니다.

| 평가 주체 | 평가 항목 | 대조군(Baseline) 승리 | 실험군(Proposed) 승리 | 무승부 |
| :--- | :--- | :---: | :---: | :---: |
| **인간 블라인드 평가** | 감정 보존성 | 20 | **62** | 68 |
| **LLM Blind A/B** | 감정 보존성 | 34 | **227** | 239 |

> 📌 **학술적 결론**: 본 연구에서 제안한 시스템은 번역문의 구조적 유창성과 정보성을 해치지 않으면서, 텍스트 뒤에 숨겨진 화자의 정서적 어조를 안정적으로 영문화하는 **'감정 보존 레이어'**로서 기능합니다.

---

## 5. 실행 가이드 및 설정

### 환경 구축
필요 라이브러리를 설치합니다:
```bash
pip install pandas numpy scipy openpyxl torch transformers librosa soundfile tqdm openai
```

DeepSeek API 호출을 위해 환경 변수를 설정합니다:
```bash
# bash (Linux / macOS)
export DEEPSEEK_API_KEY="your_api_key_here"

# PowerShell (Windows)
$env:DEEPSEEK_API_KEY="your_api_key_here"
```

### 파이프라인 순차 실행

#### 컴포넌트 A: 감정 인식 모델링 (훈련 단계)
1. 하이퍼파라미터 탐색을 통한 최적값 설정 (선택 사항):
   ```bash
   python modeling/01_optimize_hyperparams.py
   ```
2. 멀티모달 모델(오디오 + 종결어미) 및 오디오 단독 baseline 모델 학습 실행:
   ```bash
   python modeling/02_train_multimodal.py
   python modeling/03_train_audio_only.py
   ```

#### 컴포넌트 B: 번역 및 평가 파이프라인
1. 전처리 및 감정 차원(VAE) 라벨링 진행:
   ```bash
   python translation_eval/01_prepare_eval_inputs.py
   python translation_eval/02_predict_audio_ef_vae.py
   ```
2. 감정 인지형 번역 생성 및 영문 번역에서의 V/A 차원 추출:
   ```bash
   python translation_eval/03_generate_translation_audio_pred_deepseek.py
   python translation_eval/04_predict_va_vadbert.py
   ```
3. 정성 평가 및 자동 요약 실행:
   ```bash
   python translation_eval/05_evaluate_quality_deepseek_blind.py
   python translation_eval/06_compute_final_summary.py
   ```

평가 파이프라인의 전체 10단계 실행 명세는 [translation_eval/README.md](file:///c:/Project/MLP_Emotion_Translation/translation_eval/README.md)에 상세히 서술되어 있습니다.

## 정성적 번역 사례 분석 (Qualitative Translation Examples)

정량적 평가 결과는 감정 일관성(emotion consistency)의 향상을 보여주지만, 실제 번역 사례를 살펴보면 감정 정보가 번역 결과에 어떤 영향을 미치는지 더욱 직관적으로 확인할 수 있다.

아래 예시는 다음 두 가지 번역 결과를 비교한다.

* **기본 번역 (Baseline Translation)**: 감정 정보를 제공하지 않은 경우
* **감정 인식 기반 번역 (Emotion-aware Translation)**: 예측된 감정 정보를 함께 제공한 경우

### 1. 문장부호를 통한 감정 표현 강화

감정 정보를 활용한 번역은 물음표(?)나 느낌표(!)와 같은 문장부호를 보다 적절하게 사용하여, 화자의 감정 강도를 더욱 자연스럽게 반영하는 경향을 보인다.

**원문 (한국어)**

[이게 맥시멈이]

**기본 번역**

[This is the maximum.]

**감정 인식 기반 번역**

[This is the maximum?]

**분석**

감정 정보를 활용한 번역은 문장부호를 통해 화자의 감정적 반응을 보다 효과적으로 표현하며, 결과적으로 원문의 분위기를 더욱 잘 전달한다.

---

### 2. 어조 및 태도 반영

감정 정보는 단순한 의미 전달을 넘어 발화의 전반적인 어조와 태도에도 영향을 미친다. 감정 인식 기반 번역은 당황, 놀람, 실망, 흥분 등 화자의 정서를 보다 적절한 표현으로 반영하는 경우가 많다.

**원문 (한국어)**

[뭐야]

**기본 번역**

[What is it]

**감정 인식 기반 번역**

[What the...?]

**분석**

감정 인식 기반 번역은 화자의 의도와 정서를 더욱 자연스럽게 반영하여, 실제 발화 상황에 더 적합한 번역 결과를 생성한다.

---

### 3. 한국어 종결어미의 감정 정보 보존

한국어의 종결어미는 화자의 감정, 태도, 그리고 대인관계적 의미를 담고 있는 중요한 언어적 요소이다. 감정 정보를 활용함으로써 번역 모델은 이러한 종결어미의 화용적(pragmatic) 기능을 목표 언어에서도 보다 효과적으로 반영할 수 있다.

**원문 (한국어)**

[뭐 그런 분들도 계실 텐데]

**기본 번역**

[Well, there might be people like that.]

**감정 인식 기반 번역**

[Well, there might be people like that too, I suppose.]

**분석**

감정 인식 기반 번역은 종결어미에 내포된 미묘한 감정과 화자의 태도를 더욱 잘 전달하며, 원문의 의도에 가까운 번역 결과를 제공한다.

---

## 요약

위 사례들은 감정 인식 기반 번역이 다음과 같은 측면에서 개선 효과를 보였음을 보여준다.

* 감정이 반영된 문장부호 사용
* 어조 및 화자 태도의 일관성 향상
* 한국어 종결어미의 감정적·화용적 의미 보존

이러한 정성적 분석 결과는 정량적 평가 결과와도 일치하며, 의미 정확성을 유지하면서 감정 일관성을 효과적으로 향상시킬 수 있음을 보여준다.

