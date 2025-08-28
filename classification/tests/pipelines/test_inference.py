"""
추론 파이프라인을 위한 테스트 모듈입니다.
"""
import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import Mock, patch
from pathlib import Path
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project

from classification.pipelines.inference.nodes import (
    make_predictions,
    evaluate_predictions
)


class TestInferenceNodes:
    """추론 노드들을 위한 테스트 클래스"""
    
    @pytest.fixture(autouse=True)
    def setup_project(self):
        """각 테스트 전에 프로젝트 설정"""
        self.project_path = Path.cwd()
        bootstrap_project(self.project_path)
        
    def test_make_predictions_structure(self):
        """make_predictions 함수의 출력 구조 확인"""
        # Mock 모델 생성
        mock_model = Mock()
        mock_model.eval = Mock()
        mock_model.to = Mock(return_value=mock_model)
        
        # Mock 예측 결과 설정
        mock_outputs = torch.tensor([[0.8, 0.2], [0.3, 0.7], [0.9, 0.1]])
        mock_model.return_value = mock_outputs
        
        # Mock 전처리된 테스트 데이터
        mock_test_data = {
            "test_loader": Mock(),
            "image_paths": ["img1.jpg", "img2.jpg", "img3.jpg"],
            "test_size": 3
        }
        
        # Mock DataLoader 설정
        mock_test_data["test_loader"].__len__ = Mock(return_value=1)
        mock_test_data["test_loader"].__iter__ = Mock(return_value=iter([
            (torch.randn(3, 3, 64, 64), torch.tensor([0, 1, 0]))
        ]))
        
        # Mock 파라미터
        mock_parameters = {
            "model": {"device": "cpu"},
            "inference": {"threshold": 0.5}
        }
        
        try:
            # 예측 실행
            with patch('torch.no_grad'):
                predictions = make_predictions(mock_model, mock_test_data, mock_parameters)
            
            # 반환 구조 확인
            assert isinstance(predictions, dict)
            expected_keys = [
                "predictions",
                "binary_predictions", 
                "probabilities",
                "image_paths",
                "threshold"
            ]
            
            for key in expected_keys:
                assert key in predictions
                
            # 데이터 타입 확인
            assert isinstance(predictions["predictions"], np.ndarray)
            assert isinstance(predictions["binary_predictions"], np.ndarray)
            assert isinstance(predictions["probabilities"], np.ndarray)
            assert isinstance(predictions["image_paths"], list)
            assert isinstance(predictions["threshold"], float)
            
        except Exception as e:
            # GPU 관련 에러는 예상됨
            if "cuda" not in str(e).lower():
                raise e
                
    def test_evaluate_predictions_with_labels(self):
        """라벨이 있는 경우 evaluate_predictions 함수 테스트"""
        # Mock 예측 결과
        mock_predictions = {
            "predictions": np.array([0, 1, 0, 1]),
            "binary_predictions": np.array([0, 1, 0, 1]),
            "probabilities": np.array([[0.8, 0.2], [0.3, 0.7], [0.9, 0.1], [0.4, 0.6]]),
            "image_paths": ["img1.jpg", "img2.jpg", "img3.jpg", "img4.jpg"],
            "threshold": 0.5
        }
        
        # Mock 테스트 데이터 (라벨 포함)
        mock_test_data = {
            "true_labels": [0, 1, 0, 0],  # 실제 라벨
            "image_paths": ["img1.jpg", "img2.jpg", "img3.jpg", "img4.jpg"]
        }
        
        # 평가 실행
        evaluation_report = evaluate_predictions(mock_predictions, mock_test_data)
        
        # 반환 구조 확인
        assert isinstance(evaluation_report, dict)
        
        # 기본 정보 확인
        assert "total_samples" in evaluation_report
        assert "normal_predictions" in evaluation_report
        assert "defect_predictions" in evaluation_report
        assert "average_confidence" in evaluation_report
        assert "threshold" in evaluation_report
        
        # 성능 메트릭 확인 (라벨이 있으므로)
        assert "performance_metrics" in evaluation_report
        metrics = evaluation_report["performance_metrics"]
        
        expected_metrics = ["accuracy", "precision", "recall", "f1_score"]
        for metric in expected_metrics:
            assert metric in metrics
            assert 0 <= metrics[metric] <= 1
            
        # 혼동 행렬 확인
        assert "confusion_matrix" in evaluation_report
        cm = evaluation_report["confusion_matrix"]
        
        expected_cm_keys = [
            "true_normal_pred_normal",
            "true_normal_pred_defect",
            "true_defect_pred_normal", 
            "true_defect_pred_defect"
        ]
        
        for key in expected_cm_keys:
            assert key in cm
            
    def test_evaluate_predictions_without_labels(self):
        """라벨이 없는 경우 evaluate_predictions 함수 테스트"""
        # Mock 예측 결과
        mock_predictions = {
            "predictions": np.array([0, 1, 0]),
            "binary_predictions": np.array([0, 1, 0]),
            "probabilities": np.array([[0.8, 0.2], [0.3, 0.7], [0.9, 0.1]]),
            "image_paths": ["img1.jpg", "img2.jpg", "img3.jpg"],
            "threshold": 0.5
        }
        
        # Mock 테스트 데이터 (라벨 없음)
        mock_test_data = {
            "image_paths": ["img1.jpg", "img2.jpg", "img3.jpg"]
        }
        
        # 평가 실행
        evaluation_report = evaluate_predictions(mock_predictions, mock_test_data)
        
        # 반환 구조 확인
        assert isinstance(evaluation_report, dict)
        
        # 기본 정보는 있어야 함
        assert "total_samples" in evaluation_report
        assert "normal_predictions" in evaluation_report
        assert "defect_predictions" in evaluation_report
        
        # 성능 메트릭은 없어야 함 (라벨이 없으므로)
        assert "performance_metrics" not in evaluation_report or \
               evaluation_report.get("performance_metrics") is None


class TestInferencePipeline:
    """추론 파이프라인 전체 테스트 클래스"""
    
    @pytest.fixture(autouse=True)
    def setup_project(self):
        """각 테스트 전에 프로젝트 설정"""
        self.project_path = Path.cwd()
        bootstrap_project(self.project_path)
        
    def test_inference_pipeline_structure(self):
        """추론 파이프라인 구조 확인"""
        with KedroSession.create(project_path=self.project_path) as session:
            context = session.load_context()
            pipeline = context.pipelines["inference"]
            
            # 노드 개수 확인 (2개)
            assert len(pipeline.nodes) == 2
            
            # 노드 이름 확인
            node_names = [node.name for node in pipeline.nodes]
            expected_nodes = ["06_make_predictions", "07_evaluate_predictions"]
            
            for expected_node in expected_nodes:
                assert expected_node in node_names
                
    def test_inference_pipeline_dependencies(self):
        """추론 파이프라인 의존성 확인"""
        with KedroSession.create(project_path=self.project_path) as session:
            context = session.load_context()
            pipeline = context.pipelines["inference"]
            
            # 입력/출력 확인
            inputs = pipeline.all_inputs()
            outputs = pipeline.all_outputs()
            
            # 입력 확인
            assert "trained_model" in inputs
            assert "preprocessed_test_data" in inputs
            assert "parameters" in inputs
            
            # 출력 확인
            assert "predictions" in outputs
            assert "evaluation_report" in outputs
            
    def test_inference_pipeline_flow(self):
        """추론 파이프라인의 데이터 흐름 확인"""
        with KedroSession.create(project_path=self.project_path) as session:
            context = session.load_context()
            pipeline = context.pipelines["inference"]
            
            # 노드 간 연결 확인
            nodes = {node.name: node for node in pipeline.nodes}
            
            # make_predictions 노드의 출력이 evaluate_predictions의 입력인지 확인
            make_pred_node = nodes["06_make_predictions"]
            eval_node = nodes["07_evaluate_predictions"]
            
            # make_predictions의 출력
            make_pred_outputs = set(make_pred_node.outputs)
            
            # evaluate_predictions의 입력
            eval_inputs = set(eval_node.inputs)
            
            # predictions가 연결되어 있는지 확인
            assert "predictions" in make_pred_outputs
            assert "predictions" in eval_inputs
