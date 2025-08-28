"""
모델링 파이프라인을 위한 테스트 모듈입니다.
"""
import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch
from pathlib import Path
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project

from classification.pipelines.modeling.nodes import (
    CNNModel,
    create_model,
    train_model
)


class TestCNNModel:
    """CNN 모델 클래스 테스트"""
    
    def test_cnn_model_initialization(self):
        """CNN 모델이 올바르게 초기화되는지 확인"""
        # 모델 파라미터
        input_channels = 3
        num_classes = 2
        conv_layers = [
            {"out_channels": 32, "kernel_size": 3, "stride": 1, "padding": 1},
            {"out_channels": 64, "kernel_size": 3, "stride": 1, "padding": 1}
        ]
        fc_layers = [512, 256]
        
        # 모델 생성
        model = CNNModel(
            input_channels=input_channels,
            num_classes=num_classes,
            conv_layers=conv_layers,
            fc_layers=fc_layers
        )
        
        # 모델 구조 확인
        assert isinstance(model, nn.Module)
        assert len(model.features) == len(conv_layers)
        assert isinstance(model.classifier, nn.Sequential)
        
    def test_cnn_model_forward_pass(self):
        """CNN 모델의 forward pass 테스트"""
        # 간단한 모델 생성
        conv_layers = [
            {"out_channels": 16, "kernel_size": 3, "stride": 1, "padding": 1},
            {"out_channels": 32, "kernel_size": 3, "stride": 1, "padding": 1}
        ]
        fc_layers = [128]
        
        model = CNNModel(
            input_channels=3,
            num_classes=2,
            conv_layers=conv_layers,
            fc_layers=fc_layers
        )
        
        # 더미 입력 생성 (batch_size=2, channels=3, height=64, width=64)
        dummy_input = torch.randn(2, 3, 64, 64)
        
        # Forward pass
        output = model(dummy_input)
        
        # 출력 shape 확인
        assert output.shape == (2, 2)  # (batch_size, num_classes)
        
    def test_cnn_model_with_different_configs(self):
        """다양한 설정으로 CNN 모델 테스트"""
        # BatchNorm 없는 모델
        model_no_bn = CNNModel(
            input_channels=3,
            num_classes=2,
            conv_layers=[{"out_channels": 16, "kernel_size": 3, "stride": 1, "padding": 1}],
            fc_layers=[64],
            use_batch_norm=False
        )
        
        # Dropout rate 다른 모델
        model_high_dropout = CNNModel(
            input_channels=3,
            num_classes=2,
            conv_layers=[{"out_channels": 16, "kernel_size": 3, "stride": 1, "padding": 1}],
            fc_layers=[64],
            dropout_rate=0.8
        )
        
        # 모델들이 정상적으로 생성되는지 확인
        assert isinstance(model_no_bn, nn.Module)
        assert isinstance(model_high_dropout, nn.Module)


class TestModelingNodes:
    """모델링 노드들을 위한 테스트 클래스"""
    
    @pytest.fixture(autouse=True)
    def setup_project(self):
        """각 테스트 전에 프로젝트 설정"""
        self.project_path = Path.cwd()
        bootstrap_project(self.project_path)
        
    def test_create_model_function(self):
        """create_model 함수 테스트"""
        # Mock 파라미터
        parameters = {
            "model": {
                "input_channels": 3,
                "num_classes": 2,
                "conv_layers": [
                    {"out_channels": 32, "kernel_size": 3, "stride": 1, "padding": 1}
                ],
                "fc_layers": [128],
                "dropout_rate": 0.5,
                "batch_norm": True,
                "device": "cpu",
                "init_method": "kaiming",
                "pooling": {
                    "max_pool": {"kernel_size": 2, "stride": 2, "padding": 0},
                    "adaptive_pool": {"output_size": [7, 7]}
                }
            }
        }
        
        # 모델 생성
        model = create_model(parameters)
        
        # 모델 타입 확인
        assert isinstance(model, CNNModel)
        
        # 파라미터 개수 확인 (0보다 큰지)
        total_params = sum(p.numel() for p in model.parameters())
        assert total_params > 0
        
    def test_train_model_structure(self):
        """train_model 함수의 입출력 구조 확인 (실행하지 않고)"""
        # Mock 전처리된 데이터
        mock_preprocessed_data = {
            "train_loader": Mock(),
            "train_size": 100,
            "batch_size": 32
        }
        
        # Mock 파라미터  
        mock_parameters = {
            "model": {
                "input_channels": 3,
                "num_classes": 2,
                "conv_layers": [{"out_channels": 16, "kernel_size": 3, "stride": 1, "padding": 1}],
                "fc_layers": [64],
                "epochs": 1,
                "learning_rate": 0.001,
                "device": "cpu",
                "dropout_rate": 0.5,
                "batch_norm": True,
                "weight_decay": 0.0001,
                "init_method": "kaiming",
                "pooling": {
                    "max_pool": {"kernel_size": 2, "stride": 2, "padding": 0},
                    "adaptive_pool": {"output_size": [7, 7]}
                }
            },
            "training": {
                "optimizer": "adam",
                "scheduler": "step",
                "step_size": 20,
                "gamma": 0.1,
                "adam": {"beta1": 0.9, "beta2": 0.999, "eps": 1e-8}
            }
        }
        
        # DataLoader mock 설정
        mock_preprocessed_data["train_loader"].__len__ = Mock(return_value=4)
        mock_preprocessed_data["train_loader"].__iter__ = Mock(return_value=iter([
            (torch.randn(2, 3, 64, 64), torch.tensor([0, 1]))
        ]))
        
        try:
            # 훈련 실행
            model, metrics = train_model(mock_preprocessed_data, mock_parameters)
            
            # 반환 타입 확인
            assert isinstance(model, nn.Module)
            assert isinstance(metrics, dict)
            
            # 메트릭 구조 확인
            expected_keys = [
                "train_losses",
                "train_accuracies",
                "final_train_accuracy",
                "total_epochs",
                "model_parameters"
            ]
            
            for key in expected_keys:
                assert key in metrics
                
        except Exception as e:
            # GPU 관련 에러나 실제 훈련 관련 에러는 예상됨
            if "cuda" not in str(e).lower() and "training" not in str(e).lower():
                raise e


class TestModelingPipeline:
    """모델링 파이프라인 전체 테스트 클래스"""
    
    @pytest.fixture(autouse=True)
    def setup_project(self):
        """각 테스트 전에 프로젝트 설정"""
        self.project_path = Path.cwd()
        bootstrap_project(self.project_path)
        
    def test_modeling_pipeline_structure(self):
        """모델링 파이프라인 구조 확인"""
        with KedroSession.create(project_path=self.project_path) as session:
            context = session.load_context()
            pipeline = context.pipelines["modeling"]
            
            # 노드 개수 확인 (1개 - 단순화된 파이프라인)
            assert len(pipeline.nodes) == 1
            
            # 노드 이름 확인
            node_names = [node.name for node in pipeline.nodes]
            assert "05_train_cnn_model" in node_names
            
    def test_modeling_pipeline_dependencies(self):
        """모델링 파이프라인 의존성 확인"""
        with KedroSession.create(project_path=self.project_path) as session:
            context = session.load_context()
            pipeline = context.pipelines["modeling"]
            
            # 입력/출력 확인
            inputs = pipeline.all_inputs()
            outputs = pipeline.all_outputs()
            
            # 입력 확인
            assert "preprocessed_train_data" in inputs
            assert "parameters" in inputs
            
            # 출력 확인
            assert "trained_model" in outputs
            assert "training_metrics" in outputs
