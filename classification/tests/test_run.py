"""
CNN 분류 파이프라인을 위한 포괄적인 테스트 모듈입니다.
모든 파이프라인 구성요소와 통합 테스트를 포함합니다.
"""
import pytest
import os
import tempfile
import torch
import numpy as np
from pathlib import Path
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project
from kedro.io import DataCatalog
from kedro.config import OmegaConfigLoader


class TestCNNPipelineIntegration:
    """CNN 분류 파이프라인 통합 테스트 클래스"""
    
    @pytest.fixture(autouse=True)
    def setup_project(self):
        """각 테스트 전에 프로젝트 설정"""
        self.project_path = Path.cwd()
        bootstrap_project(self.project_path)
        
    def test_pipeline_exists(self):
        """파이프라인이 존재하는지 확인"""
        with KedroSession.create(project_path=self.project_path) as session:
            context = session.load_context()
            # Kedro 1.0.0에서는 registry를 통해 파이프라인에 접근
            from classification.pipeline_registry import create_pipelines
            pipelines = create_pipelines()
            
            # 기본 파이프라인들이 존재하는지 확인
            assert "__default__" in pipelines
            assert "data_processing" in pipelines
            assert "modeling" in pipelines  
            assert "inference" in pipelines
            
    def test_pipeline_structure(self):
        """파이프라인 구조가 올바른지 확인"""
        with KedroSession.create(project_path=self.project_path) as session:
            context = session.load_context()
            from classification.pipeline_registry import create_pipelines
            pipelines = create_pipelines()
            default_pipeline = pipelines["__default__"]
            
            # 노드 개수 확인 (7개 노드)
            assert len(default_pipeline.nodes) == 7
            
            # 노드 이름들 확인
            node_names = [node.name for node in default_pipeline.nodes]
            expected_nodes = [
                "01_load_raw_training_data",
                "02_load_raw_test_data", 
                "03_preprocess_training_data",
                "04_preprocess_test_data",
                "05_train_cnn_model",
                "06_make_predictions",
                "07_evaluate_predictions"
            ]
            
            for expected_node in expected_nodes:
                assert expected_node in node_names
                
    def test_data_catalog_configuration(self):
        """데이터 카탈로그 설정이 올바른지 확인"""
        with KedroSession.create(project_path=self.project_path) as session:
            context = session.load_context()
            catalog = context.catalog
            
            # 필수 데이터셋들이 정의되어 있는지 확인
            expected_datasets = [
                "raw_train_data",
                "raw_test_data", 
                "preprocessed_train_data",
                "preprocessed_test_data",
                "trained_model",
                "training_metrics",
                "predictions",
                "evaluation_report"
            ]
            
            for dataset in expected_datasets:
                assert dataset in catalog._datasets
                
    def test_parameters_configuration(self):
        """파라미터 설정이 올바른지 확인"""
        config_loader = OmegaConfigLoader(conf_source=str(self.project_path / "conf"))
        parameters = config_loader["parameters"]
        
        # 필수 파라미터 섹션들 확인
        assert "data_processing" in parameters
        assert "model" in parameters
        assert "training" in parameters
        assert "inference" in parameters
        
        # 데이터 처리 파라미터 확인
        data_params = parameters["data_processing"]
        assert "image_size" in data_params
        assert "batch_size" in data_params
        assert len(data_params["image_size"]) == 2
        
        # 모델 파라미터 확인  
        model_params = parameters["model"]
        assert "input_channels" in model_params
        assert "num_classes" in model_params
        assert "conv_layers" in model_params
        assert "fc_layers" in model_params
        
    def test_data_directory_structure(self):
        """데이터 디렉토리 구조가 올바른지 확인"""
        data_path = self.project_path / "data"
        
        # 필수 데이터 디렉토리들 확인
        expected_dirs = [
            "01_raw",
            "05_model_input", 
            "06_models",
            "07_model_output",
            "08_reporting"
        ]
        
        for dir_name in expected_dirs:
            assert (data_path / dir_name).exists()
            
        # 원시 데이터 구조 확인
        raw_path = data_path / "01_raw"
        if (raw_path / "train_data").exists():
            train_path = raw_path / "train_data"
            assert (train_path / "good").exists()
            assert (train_path / "defective").exists()
            
        if (raw_path / "test_data").exists():
            test_path = raw_path / "test_data"
            assert (test_path / "good").exists()
            assert (test_path / "defective").exists()


class TestPipelineExecution:
    """파이프라인 실행 테스트 클래스"""
    
    @pytest.fixture(autouse=True)
    def setup_project(self):
        """각 테스트 전에 프로젝트 설정"""
        self.project_path = Path.cwd()
        bootstrap_project(self.project_path)
        
    @pytest.mark.slow
    def test_full_pipeline_execution(self):
        """전체 파이프라인이 성공적으로 실행되는지 확인 (느린 테스트)"""
        with KedroSession.create(project_path=self.project_path) as session:
            try:
                # 전체 파이프라인 실행
                session.run()
                
                # 결과 파일들이 생성되었는지 확인
                data_path = self.project_path / "data"
                
                # 모델 파일 확인
                model_file = data_path / "06_models" / "cnn_model.pkl"
                assert model_file.exists()
                
                # 예측 결과 파일 확인  
                predictions_file = data_path / "07_model_output" / "predictions.pkl"
                assert predictions_file.exists()
                
                # 평가 보고서 확인
                evaluation_file = data_path / "08_reporting" / "evaluation_report.json"
                assert evaluation_file.exists()
                
                # 훈련 메트릭 확인
                training_metrics_file = data_path / "08_reporting" / "training_metrics.json"
                assert training_metrics_file.exists()
                
            except Exception as e:
                pytest.fail(f"파이프라인 실행 실패: {str(e)}")
                
    def test_individual_pipeline_execution(self):
        """개별 파이프라인들이 실행되는지 확인"""
        with KedroSession.create(project_path=self.project_path) as session:
            context = session.load_context()
            
            # 데이터 처리 파이프라인만 실행
            try:
                session.run(pipeline_name="data_processing")
            except Exception as e:
                pytest.fail(f"데이터 처리 파이프라인 실행 실패: {str(e)}")


class TestOutputValidation:
    """파이프라인 출력 검증 테스트 클래스"""
    
    @pytest.fixture(autouse=True) 
    def setup_project(self):
        """각 테스트 전에 프로젝트 설정"""
        self.project_path = Path.cwd()
        bootstrap_project(self.project_path)
        
    def test_evaluation_report_structure(self):
        """평가 보고서 구조가 올바른지 확인"""
        evaluation_file = self.project_path / "data" / "08_reporting" / "evaluation_report.json"
        
        if evaluation_file.exists():
            import json
            with open(evaluation_file, 'r') as f:
                report = json.load(f)
                
            # 필수 키들 확인
            expected_keys = [
                "total_samples",
                "normal_predictions", 
                "defect_predictions",
                "average_confidence",
                "threshold",
                "prediction_summary"
            ]
            
            for key in expected_keys:
                assert key in report
                
            # 성능 메트릭이 있다면 확인
            if "performance_metrics" in report:
                metrics = report["performance_metrics"]
                expected_metrics = ["accuracy", "precision", "recall", "f1_score"]
                
                for metric in expected_metrics:
                    assert metric in metrics
                    assert 0 <= metrics[metric] <= 1  # 0과 1 사이 값
                    
    def test_training_metrics_structure(self):
        """훈련 메트릭 구조가 올바른지 확인"""  
        metrics_file = self.project_path / "data" / "08_reporting" / "training_metrics.json"
        
        if metrics_file.exists():
            import json
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
                
            # 필수 키들 확인
            expected_keys = [
                "train_losses",
                "train_accuracies", 
                "final_train_accuracy",
                "total_epochs",
                "model_parameters"
            ]
            
            for key in expected_keys:
                assert key in metrics
                
            # 값들의 유효성 확인
            assert isinstance(metrics["train_losses"], list)
            assert isinstance(metrics["train_accuracies"], list)
            assert 0 <= metrics["final_train_accuracy"] <= 1
            assert metrics["total_epochs"] > 0
            assert metrics["model_parameters"] > 0
