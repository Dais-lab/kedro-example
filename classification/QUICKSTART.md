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
  epochs: 5 # 2에서 5로 변경

training:
  learning_rate: 0.0001 # 0.001에서 0.0001로 변경
```

```bash
# 다시 실행
kedro run
```

## 🎉 완료!

이제 Kedro의 기본 사용법을 익혔습니다. 자세한 내용은 [README.md](README.md)를 참조하세요.

# classification

[![Powered by Kedro](https://img.shields.io/badge/powered_by-kedro-ffc900?logo=kedro)](https://kedro.org)

## Overview

This is your new Kedro project, which was generated using `kedro 1.0.0`.

Take a look at the [Kedro documentation](https://docs.kedro.org) to get started.

## Rules and guidelines

In order to get the best out of the template:

- Don't remove any lines from the `.gitignore` file we provide
- Make sure your results can be reproduced by following a data engineering convention
- Don't commit data to your repository
- Don't commit any credentials or your local configuration to your repository. Keep all your credentials and local configuration in `conf/local/`

## How to install dependencies

Declare any dependencies in `requirements.txt` for `pip` installation.

To install them, run:

```
pip install -r requirements.txt
```

## How to run your Kedro pipeline

You can run your Kedro project with:

```
kedro run
```

## How to test your Kedro project

Have a look at the file `tests/test_run.py` for instructions on how to write your tests. You can run your tests as follows:

```
pytest
```

You can configure the coverage threshold in your project's `pyproject.toml` file under the `[tool.coverage.report]` section.

## Project dependencies

To see and update the dependency requirements for your project use `requirements.txt`. You can install the project requirements with `pip install -r requirements.txt`.

[Further information about project dependencies](https://docs.kedro.org/en/stable/kedro_project_setup/dependencies.html#project-specific-dependencies)

## How to work with Kedro and notebooks

> Note: Using `kedro jupyter` or `kedro ipython` to run your notebook provides these variables in scope: `context`, 'session', `catalog`, and `pipelines`.
>
> Jupyter, JupyterLab, and IPython are already included in the project requirements by default, so once you have run `pip install -r requirements.txt` you will not need to take any extra steps before you use them.

### Jupyter

To use Jupyter notebooks in your Kedro project, you need to install Jupyter:

```
pip install jupyter
```

After installing Jupyter, you can start a local notebook server:

```
kedro jupyter notebook
```

### JupyterLab

To use JupyterLab, you need to install it:

```
pip install jupyterlab
```

You can also start JupyterLab:

```
kedro jupyter lab
```

### IPython

And if you want to run an IPython session:

```
kedro ipython
```

### How to ignore notebook output cells in `git`

To automatically strip out all output cell contents before committing to `git`, you can use tools like [`nbstripout`](https://github.com/kynan/nbstripout). For example, you can add a hook in `.git/config` with `nbstripout --install`. This will run `nbstripout` before anything is committed to `git`.

> _Note:_ Your output cells will be retained locally.

## Package your Kedro project

[Further information about building project documentation and packaging your project](https://docs.kedro.org/en/stable/tutorial/package_a_project.html)
