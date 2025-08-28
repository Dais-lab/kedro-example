#!/usr/bin/env python3
"""
CNN 분류 파이프라인 테스트 실행 스크립트입니다.
다양한 테스트 옵션을 제공합니다.
"""
import subprocess
import sys
import argparse
from pathlib import Path


def run_command(cmd, description):
    """명령어를 실행하고 결과를 출력합니다."""
    print(f"\n{'='*60}")
    print(f"🧪 {description}")
    print(f"{'='*60}")
    print(f"실행 명령어: {' '.join(cmd)}")
    print("-" * 60)
    
    try:
        result = subprocess.run(cmd, capture_output=False, text=True, check=True)
        print(f"\n✅ {description} 성공!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {description} 실패!")
        print(f"오류 코드: {e.returncode}")
        return False
    except Exception as e:
        print(f"\n❌ 예상치 못한 오류: {str(e)}")
        return False


def main():
    parser = argparse.ArgumentParser(description="CNN 분류 파이프라인 테스트 실행기")
    parser.add_argument("--all", action="store_true", help="모든 테스트 실행")
    parser.add_argument("--unit", action="store_true", help="단위 테스트만 실행")
    parser.add_argument("--integration", action="store_true", help="통합 테스트만 실행")
    parser.add_argument("--slow", action="store_true", help="느린 테스트 포함")
    parser.add_argument("--coverage", action="store_true", help="커버리지 포함")
    parser.add_argument("--verbose", "-v", action="store_true", help="상세 출력")
    parser.add_argument("--file", type=str, help="특정 테스트 파일 실행")
    
    args = parser.parse_args()
    
    # 기본 pytest 명령어
    base_cmd = ["python", "-m", "pytest"]
    
    if args.verbose:
        base_cmd.append("-v")
    
    success_count = 0
    total_count = 0
    
    print("🚀 CNN 분류 파이프라인 테스트 시작")
    print(f"📁 작업 디렉토리: {Path.cwd()}")
    
    # 특정 파일 테스트
    if args.file:
        cmd = base_cmd + [f"tests/{args.file}"]
        total_count += 1
        if run_command(cmd, f"특정 파일 테스트: {args.file}"):
            success_count += 1
    
    # 단위 테스트
    elif args.unit:
        test_files = [
            "tests/pipelines/test_data_processing.py::TestDataProcessingNodes",
            "tests/pipelines/test_modeling.py::TestCNNModel",
            "tests/pipelines/test_modeling.py::TestModelingNodes",
            "tests/pipelines/test_inference.py::TestInferenceNodes"
        ]
        
        for test_file in test_files:
            cmd = base_cmd + [test_file]
            total_count += 1
            if run_command(cmd, f"단위 테스트: {test_file.split('::')[-1]}"):
                success_count += 1
    
    # 통합 테스트
    elif args.integration:
        test_files = [
            "tests/test_run.py::TestCNNPipelineIntegration",
            "tests/pipelines/test_data_processing.py::TestDataProcessingPipeline",
            "tests/pipelines/test_modeling.py::TestModelingPipeline",
            "tests/pipelines/test_inference.py::TestInferencePipeline"
        ]
        
        for test_file in test_files:
            cmd = base_cmd + [test_file]
            if args.slow:
                cmd.extend(["-m", "not slow"])
            total_count += 1
            if run_command(cmd, f"통합 테스트: {test_file.split('::')[-1]}"):
                success_count += 1
    
    # 모든 테스트
    elif args.all:
        cmd = base_cmd + ["tests/"]
        if not args.slow:
            cmd.extend(["-m", "not slow"])
        if args.coverage:
            cmd.extend(["--cov=src/classification", "--cov-report=html", "--cov-report=term"])
        
        total_count += 1
        if run_command(cmd, "전체 테스트 스위트"):
            success_count += 1
    
    # 기본: 빠른 테스트들만
    else:
        cmd = base_cmd + ["tests/", "-m", "not slow"]
        total_count += 1
        if run_command(cmd, "빠른 테스트들 (느린 테스트 제외)"):
            success_count += 1
    
    # 결과 요약
    print(f"\n{'='*60}")
    print("📊 테스트 결과 요약")
    print(f"{'='*60}")
    print(f"✅ 성공: {success_count}")
    print(f"❌ 실패: {total_count - success_count}")
    print(f"📈 성공률: {success_count/total_count*100:.1f}%" if total_count > 0 else "📈 성공률: N/A")
    
    if args.coverage and success_count > 0:
        print(f"\n📋 커버리지 보고서가 htmlcov/ 디렉토리에 생성되었습니다.")
        print("   브라우저에서 htmlcov/index.html 파일을 열어 확인하세요.")
    
    # 추가 도움말
    if not any([args.all, args.unit, args.integration, args.file]):
        print(f"\n💡 추가 옵션:")
        print(f"   --all           모든 테스트 실행")
        print(f"   --unit          단위 테스트만 실행")
        print(f"   --integration   통합 테스트만 실행")
        print(f"   --slow          느린 테스트 포함")
        print(f"   --coverage      커버리지 측정")
        print(f"   --file <파일명>  특정 테스트 파일 실행")
        print(f"   --verbose       상세 출력")
    
    return success_count == total_count


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
