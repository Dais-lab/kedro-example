# CNN 이미지 분류 Kedro 파이프라인

> **Kedro 학습을 위한 실습용 CNN 이미지 분류 파이프라인**  
> Kedro 프레임워크를 익히기 위한 예제 프로젝트입니다.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Kedro](https://img.shields.io/badge/Kedro-1.0.0-orange.svg)](https://kedro.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-Latest-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 목차

- [프로젝트 개요](#-프로젝트-개요)
- [주요 특징](#-주요-특징)
- [파이프라인 구조](#-파이프라인-구조)
- [설치 및 설정](#-설치-및-설정)
- [사용법](#-사용법)
- [데이터 구조](#-데이터-구조)
- [파이프라인 상세](#-파이프라인-상세)
- [테스트](#-테스트)
- [설정 파일](#-설정-파일)
- [결과 해석](#-결과-해석)
- [Kedro 핵심 개념](#-kedro-핵심-개념)
- [문제 해결](#-문제-해결)
- [기여 방법](#-기여-방법)

## 프로젝트 개요

이 프로젝트는 **이미지 분류를 위한 CNN(Convolutional Neural Network) 파이프라인**을 Kedro 프레임워크로 구현한 예제입니다.
Good/Defective 이진 분류 문제를 해결하며, Kedro의 핵심 개념들을 실습할 수 있도록 설계되었습니다.

### 학습 목표

- Kedro 프레임워크의 기본 구조 이해
- 데이터 파이프라인 설계 및 구현 경험
- ML 모델 훈련 및 평가 자동화
- 재현 가능한 ML 워크플로우 구축

## 주요 특징

### **기술 스택**

- **프레임워크**: Kedro 1.0.0
- **딥러닝**: PyTorch + torchvision
- **데이터 처리**: NumPy, Pillow
- **평가**: scikit-learn
- **시각화**: Matplotlib
- **테스트**: pytest

### **Kedro 핵심 기능 활용**

- ✅ **모듈화된 파이프라인** (데이터 처리, 모델링, 추론)
- ✅ **데이터 카탈로그** (자동 데이터 저장/로딩)
- ✅ **파라미터 관리** (YAML 기반 설정)
- ✅ **파이프라인 시각화** (Kedro Viz)
- ✅ **재현 가능한 실험**
- ✅ **포괄적인 테스트**

### **파이프라인 단계**

1. **데이터 로딩**: 훈련/테스트 이미지 로드
2. **데이터 전처리**: 이미지 리사이징, 정규화, 증강
3. **모델 훈련**: CNN 모델 훈련 (전체 데이터 사용)
4. **추론**: 테스트 데이터에 대한 예측
5. **평가**: 성능 지표 계산 및 보고서 생성

## 🚀 설치 및 설정

### 1️⃣ **환경 요구사항**

- Python 3.8+
- CUDA (GPU 사용 시, 선택사항)

### 2️⃣ **프로젝트 클론**

```bash
git clone <repository-url>
cd classification
```

### 3️⃣ **의존성 설치**

```bash
pip install -r requirements.txt
```

### 4️⃣ **데이터 준비**

데이터를 다음 구조로 배치하세요:

```
data/01_raw/
├── train_data/
│   ├── good/          # 정상 이미지들
│   └── defective/     # 결함 이미지들
└── test_data/
    ├── good/          # 정상 테스트 이미지들
    └── defective/     # 결함 테스트 이미지들
```

## 사용법

### **전체 파이프라인 실행**

```bash
# 전체 파이프라인 실행
kedro run

# 특정 파이프라인만 실행
kedro run --pipeline data_processing
kedro run --pipeline modeling
kedro run --pipeline inference
```

### **파이프라인 시각화**

```bash
# Kedro Viz 실행 (웹 브라우저에서 확인)
kedro viz
```

### **테스트 실행**

```bash
# 간단한 테스트
python run_tests.py --file test_simple.py

# 전체 테스트
python run_tests.py --all

# 커버리지 포함
python run_tests.py --all --coverage
```

## 📁 데이터 구조

### **Kedro 데이터 레이어**

```
data/
├── 01_raw/              # 원시 데이터
├── 05_model_input/      # 전처리된 데이터
├── 06_models/           # 훈련된 모델
├── 07_model_output/     # 예측 결과
└── 08_reporting/        # 평가 보고서
```

### **출력 파일들**

- `06_models/cnn_model.pkl`: 훈련된 CNN 모델
- `07_model_output/predictions.pkl`: 예측 결과
- `08_reporting/evaluation_report.json`: 성능 평가 보고서
- `08_reporting/training_metrics.json`: 훈련 메트릭

## 🔍 파이프라인 상세

### **1. 데이터 처리 (data_processing)**

```python
# 주요 노드들
- load_raw_data()          # 훈련 데이터 로드
- load_test_data()         # 테스트 데이터 로드
- preprocess_data()        # 훈련 데이터 전처리
- preprocess_test_data()   # 테스트 데이터 전처리
```

**특징:**

- 이미지 리사이징 (256×256)
- 데이터 증강 (회전, 플리핑, 색상 조정)
- 정규화 (ImageNet 통계 사용)
- PyTorch DataLoader 생성

### **2. 모델링 (modeling)**

```python
# 주요 노드들
- train_model()           # CNN 모델 훈련
```

**CNN 모델 구조:**

- **입력**: 3채널 256×256 이미지
- **Conv 레이어**: 3개 (32→64→128 채널)
- **FC 레이어**: 2개 (512→256 뉴런)
- **출력**: 2클래스 (Good/Defective)
- **정규화**: BatchNorm + Dropout
- **활성화**: ReLU

### **3. 추론 (inference)**

```python
# 주요 노드들
- make_predictions()      # 예측 수행
- evaluate_predictions()  # 성능 평가
```

**평가 지표:**

- Accuracy, Precision, Recall, F1-Score
- 클래스별 성능 지표
- Confusion Matrix
- 신뢰도 분석

## 테스트

### **테스트 구조**

```
tests/
├── test_simple.py                    # 기본 작동 테스트 ✅
├── test_run.py                       # 통합 테스트
├── pipelines/
│   ├── test_data_processing.py       # 데이터 처리 테스트
│   ├── test_modeling.py              # 모델링 테스트
│   └── test_inference.py             # 추론 테스트
├── pytest.ini                       # pytest 설정
└── run_tests.py                      # 테스트 실행 스크립트
```

### **테스트 실행 옵션**

```bash
# 권장: 간단한 테스트
python run_tests.py --file test_simple.py

# 단위 테스트
python run_tests.py --unit

# 통합 테스트
python run_tests.py --integration

# 전체 테스트 (느림)
python run_tests.py --all --slow
```

## ⚙️ 설정 파일

### **Parameters (conf/base/parameters.yml)**

```yaml
# 데이터 처리 설정
data_processing:
  image_size: [256, 256]
  batch_size: 32
  num_workers: 0

# 모델 설정
model:
  input_channels: 3
  num_classes: 2
  conv_layers: [...]
  fc_layers: [512, 256]
  epochs: 2

# 훈련 설정
training:
  optimizer: "adam"
  learning_rate: 0.001
  scheduler: "step"
```

### **Data Catalog (conf/base/catalog.yml)**

```yaml
# 자동 데이터 저장/로딩 설정
trained_model:
  type: kedro_datasets.pickle.PickleDataset
  filepath: data/06_models/cnn_model.pkl

evaluation_report:
  type: kedro_datasets.json.JSONDataset
  filepath: data/08_reporting/evaluation_report.json
```

## 결과 해석

### **성능 지표 예시**

```json
{
  "performance_metrics": {
    "accuracy": 0.7684,
    "precision": 0.7853,
    "recall": 0.7684,
    "f1_score": 0.7099
  },
  "confusion_matrix": {
    "true_normal_pred_normal": 594,
    "true_normal_pred_defect": 8,
    "true_defect_pred_normal": 184,
    "true_defect_pred_defect": 43
  }
}
```

## 📚 Kedro 핵심 개념

### **1. 노드 (Nodes)**

```python
def load_raw_data(data_path: str) -> Dict[str, Any]:
    """데이터 로딩 노드"""
    # 데이터 로딩 로직
    return {"image_paths": paths, "labels": labels}
```

### **2. 파이프라인 (Pipeline)**

```python
def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(func=load_raw_data, inputs="params:data_path",
             outputs="raw_train_data", name="01_load_raw_training_data"),
        # ... 더 많은 노드들
    ])
```

### **3. 데이터 카탈로그 (Data Catalog)**

- 자동 데이터 저장/로딩
- 다양한 데이터 형식 지원
- 버전 관리 가능

### **4. 파라미터 관리**

- YAML 기반 설정
- 환경별 설정 가능
- 실험 재현성 보장

## 학습 가이드

### **초보자를 위한 단계별 학습**

#### **Step 1: Kedro 기본 이해**

1. `kedro run` 실행해보기
2. `kedro viz`로 파이프라인 시각화
3. 로그 메시지 관찰하기

#### **Step 2: 설정 파일 수정**

1. `conf/base/parameters.yml`에서 배치 크기 변경
2. 에포크 수 조정
3. 파이프라인 재실행

#### **Step 3: 코드 구조 탐색**

1. `src/classification/pipelines/` 디렉토리 탐색
2. 각 노드 함수들 살펴보기
3. 데이터 흐름 이해하기

#### **Step 4: 커스터마이징**

1. 새로운 평가 지표 추가
2. 모델 구조 변경
3. 데이터 증강 기법 추가

### **추가 학습 자료**

- [Kedro 공식 문서](https://kedro.org)
- [PyTorch 튜토리얼](https://pytorch.org/tutorials/)
- [MLOps 모범 사례](https://ml-ops.org/)

### **참고 자료**

- [Kedro 공식 문서](https://kedro.org)
- [PyTorch 공식 문서](https://pytorch.org)
- [프로젝트 위키](링크)

---

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

또한, 이 프로젝트는 Apache License 2.0을 따르는 [Kedro](https://github.com/kedro-org/kedro) 프레임워크를 기반으로 만들어졌습니다.

---

**🎉 Happy Learning with Kedro! 🎉**
