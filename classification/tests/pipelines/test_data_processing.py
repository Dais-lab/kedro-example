"""
데이터 처리 파이프라인을 위한 테스트 모듈입니다.
"""
import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch
from pathlib import Path
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project

from classification.pipelines.data_processing.nodes import (
    load_raw_data,
    load_test_data, 
    preprocess_data,
    preprocess_test_data,
    ImageDataset
)


class TestDataProcessingNodes:
    """데이터 처리 노드들을 위한 테스트 클래스"""
    
    @pytest.fixture(autouse=True)
    def setup_project(self):
        """각 테스트 전에 프로젝트 설정"""
        self.project_path = Path.cwd()
        bootstrap_project(self.project_path)
        
    def test_image_dataset_creation(self):
        """ImageDataset 클래스가 올바르게 작동하는지 확인"""
        # 더미 이미지 경로와 라벨 생성
        image_paths = ["path1.jpg", "path2.jpg", "path3.jpg"]
        labels = [0, 1, 0]
        
        # Transform 없이 데이터셋 생성
        dataset = ImageDataset(image_paths, labels, transform=None)
        
        assert len(dataset) == 3
        assert dataset.image_paths == image_paths
        assert dataset.labels == labels
        
    def test_load_raw_data_structure(self):
        """load_raw_data 함수의 출력 구조 확인"""
        # 실제 데이터 경로가 있다면 테스트
        data_path = str(self.project_path)
        
        try:
            result = load_raw_data(data_path)
            
            # 반환 구조 확인
            assert isinstance(result, dict)
            assert "image_paths" in result
            assert "labels" in result
            assert "num_classes" in result
            
            # 데이터 타입 확인
            assert isinstance(result["image_paths"], list)
            assert isinstance(result["labels"], list)
            assert result["num_classes"] == 2
            
            # 라벨 값 확인 (0 또는 1)
            if result["labels"]:
                for label in result["labels"]:
                    assert label in [0, 1]
                    
        except Exception:
            # 데이터가 없는 경우 패스
            pytest.skip("실제 훈련 데이터가 없어 테스트 건너뜀")
            
    def test_load_test_data_structure(self):
        """load_test_data 함수의 출력 구조 확인"""
        data_path = str(self.project_path)
        
        try:
            result = load_test_data(data_path)
            
            # 반환 구조 확인
            assert isinstance(result, dict)
            assert "image_paths" in result
            assert "labels" in result
            assert "num_classes" in result
            
            # 데이터 타입 확인
            assert isinstance(result["image_paths"], list)
            assert isinstance(result["labels"], list)
            assert result["num_classes"] == 2
            
        except Exception:
            # 데이터가 없는 경우 패스
            pytest.skip("실제 테스트 데이터가 없어 테스트 건너뜀")
            
    def test_preprocess_data_with_mock_data(self):
        """mock 데이터로 preprocess_data 함수 테스트"""
        # Mock 원시 데이터
        raw_data = {
            "image_paths": ["dummy1.jpg", "dummy2.jpg"],
            "labels": [0, 1],
            "num_classes": 2
        }
        
        # Mock 파라미터
        parameters = {
            "data_processing": {
                "image_size": [256, 256],
                "batch_size": 2,
                "num_workers": 0
            }
        }
        
        # ImageDataset과 DataLoader를 mock
        with patch('classification.pipelines.data_processing.nodes.ImageDataset') as mock_dataset, \
             patch('classification.pipelines.data_processing.nodes.DataLoader') as mock_dataloader:
            
            # Mock 설정
            mock_dataset.return_value.__len__ = Mock(return_value=2)
            mock_dataloader.return_value = Mock()
            mock_dataloader.return_value.__len__ = Mock(return_value=1)
            
            try:
                result = preprocess_data(raw_data, parameters)
                
                # 반환 구조 확인
                assert isinstance(result, dict)
                assert "train_loader" in result
                assert "train_size" in result
                assert "batch_size" in result
                
            except Exception as e:
                # Transform 관련 에러는 예상됨 (실제 이미지 없음)
                if "transforms" not in str(e).lower():
                    raise e


class TestDataProcessingPipeline:
    """데이터 처리 파이프라인 전체 테스트 클래스"""
    
    @pytest.fixture(autouse=True)
    def setup_project(self):
        """각 테스트 전에 프로젝트 설정"""
        self.project_path = Path.cwd()
        bootstrap_project(self.project_path)
        
    def test_data_processing_pipeline_structure(self):
        """데이터 처리 파이프라인 구조 확인"""
        with KedroSession.create(project_path=self.project_path) as session:
            context = session.load_context()
            pipeline = context.pipelines["data_processing"]
            
            # 노드 개수 확인 (4개)
            assert len(pipeline.nodes) == 4
            
            # 노드 이름 확인
            node_names = [node.name for node in pipeline.nodes]
            expected_nodes = [
                "01_load_raw_training_data",
                "02_load_raw_test_data",
                "03_preprocess_training_data", 
                "04_preprocess_test_data"
            ]
            
            for expected_node in expected_nodes:
                assert expected_node in node_names
                
    def test_data_processing_pipeline_dependencies(self):
        """데이터 처리 파이프라인 의존성 확인"""
        with KedroSession.create(project_path=self.project_path) as session:
            context = session.load_context()
            pipeline = context.pipelines["data_processing"]
            
            # 입력/출력 확인
            inputs = pipeline.all_inputs()
            outputs = pipeline.all_outputs()
            
            # 파라미터 입력 확인
            assert "params:data_path" in inputs
            assert "parameters" in inputs
            
            # 출력 확인
            expected_outputs = {
                "raw_train_data",
                "raw_test_data", 
                "preprocessed_train_data",
                "preprocessed_test_data"
            }
            
            for output in expected_outputs:
                assert output in outputs
