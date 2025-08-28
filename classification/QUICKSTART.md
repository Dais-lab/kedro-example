# ⚡ Quick Start Guide

> **5분 만에 Kedro CNN 파이프라인 실행하기**

## 1단계: 환경 설정

```bash
# 1. 프로젝트 클론
git clone <repository-url>
cd classification

# 2. 의존성 설치
pip install -r requirements.txt

# 3. 데이터 구조 확인
ls -la data/01_raw/
```

## 2단계: 파이프라인 실행

```bash
# 전체 파이프라인 실행 (약 2-3분 소요)
kedro run
```

## 3단계: 결과 확인

```bash
# 1. 파이프라인 시각화
kedro viz

# 2. 결과 파일 확인
ls -la data/08_reporting/
cat data/08_reporting/evaluation_report.json
```

## 4단계: 테스트 실행

```bash
# 간단한 테스트
python run_tests.py --file test_simple.py
```

## 5단계: 설정 변경해보기

```yaml
# conf/base/parameters.yml 수정
model:
  epochs: 5  # 2에서 5로 변경

training:
  learning_rate: 0.0001  # 0.001에서 0.0001로 변경
```

```bash
# 다시 실행
kedro run
```

## 🎉 완료!

이제 Kedro의 기본 사용법을 익혔습니다. 자세한 내용은 [README.md](README.md)를 참조하세요.
