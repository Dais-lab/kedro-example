"""Data processing nodes for CNN classification."""
import os
import logging
from typing import Dict, List, Tuple, Any
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


logger = logging.getLogger(__name__)


class ImageDataset(Dataset):
    """Custom dataset for loading images and labels."""
    
    def __init__(self, image_paths: List[str], labels: List[int], transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label


# 원시 데이터를 로드하고 기본적인 데이터 구조를 생성합니다
def load_raw_data(data_path: str) -> Dict[str, Any]:
    """Load raw image data from directory structure.
    
    Args:
        data_path: Path to the data directory
        
    Returns:
        Dictionary containing image paths and labels
    """
    logger.info("=" * 50)  # Task start separator
    logger.info(f"Loading raw data from {data_path}")
    logger.info("=" * 50) 
    
    image_paths = []
    labels = []
    
    # Load training data
    train_data_path = os.path.join(data_path, "01_raw", "train_data")
    
    # Load good (normal) images - label 0
    good_path = os.path.join(train_data_path, "good")
    if os.path.exists(good_path):
        for img_file in os.listdir(good_path):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(good_path, img_file))
                labels.append(0)  # Normal class
    
    # Load defective images - label 1
    defective_path = os.path.join(train_data_path, "defective")
    if os.path.exists(defective_path):
        for img_file in os.listdir(defective_path):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(defective_path, img_file))
                labels.append(1)  # Defective class
    
    logger.info(f"Loaded {len(image_paths):,} training images")
    logger.info(f"Normal: {labels.count(0):,}, Defective: {labels.count(1):,}")
    
    return {
        "image_paths": image_paths,
        "labels": labels,
        "num_classes": 2
    }


# 테스트 데이터를 로드합니다
def load_test_data(data_path: str) -> Dict[str, Any]:
    """Load labeled test image data from directory structure.
    
    Args:
        data_path: Path to the data directory
        
    Returns:
        Dictionary containing test image paths and labels
    """
    logger.info("=" * 50)  # Task start separator
    logger.info(f"Loading labeled test data from {data_path}")
    logger.info("=" * 50)
    
    image_paths = []
    labels = []
    test_data_path = os.path.join(data_path, "01_raw", "test_data")
    
    # Load good (normal) images - label 0
    good_path = os.path.join(test_data_path, "good")
    if os.path.exists(good_path):
        for img_file in os.listdir(good_path):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(good_path, img_file))
                labels.append(0)  # Normal class
    
    # Load defective images - label 1
    defective_path = os.path.join(test_data_path, "defective")
    if os.path.exists(defective_path):
        for img_file in os.listdir(defective_path):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(defective_path, img_file))
                labels.append(1)  # Defective class
    
    logger.info(f"Loaded {len(image_paths):,} test images")
    logger.info(f"Normal: {labels.count(0):,}, Defective: {labels.count(1):,}")
    
    return {
        "image_paths": image_paths,
        "labels": labels,
        "num_classes": 2
    }


# 이미지 데이터를 CNN 모델에 적합한 형태로 전처리합니다
def preprocess_data(
    raw_data: Dict[str, Any], 
    parameters: Dict[str, Any]
) -> Dict[str, Any]:
    """Preprocess image data for CNN model.
    
    Args:
        raw_data: Raw data dictionary with image paths and labels
        parameters: Processing parameters
        
    Returns:
        Preprocessed data dictionary
    """
    logger.info("=" * 50)  # Task start separator
    logger.info("Starting data preprocessing for full dataset training")
    logger.info("=" * 50)
    
    image_size = parameters["data_processing"]["image_size"]
    batch_size = parameters["data_processing"]["batch_size"]
    num_workers = parameters["data_processing"]["num_workers"]
    
    # Log key preprocessing parameters
    logger.info(f"Image size: {image_size[0]}x{image_size[1]}")
    logger.info(f"Batch size: {batch_size:,}")
    logger.info(f"Number of workers: {num_workers}")
    logger.info("Using full dataset for training (no validation split)")
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])  # ImageNet stats
    ])
    
    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    image_paths = raw_data["image_paths"]
    labels = raw_data["labels"]
    
    # Use full dataset for training (no validation split)
    total_samples = len(image_paths)
    
    logger.info(f"Total training samples: {total_samples:,}")
    logger.info(f"Normal: {labels.count(0):,}, Defect: {labels.count(1):,}")
    
    # Create full training dataset with augmentation
    train_dataset = ImageDataset(image_paths, labels, transform=train_transform)
    
    # Create data loader for full dataset
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers
    )
    
    logger.info(f"Created DataLoader with {len(train_loader):,} batches")
    logger.info("=" * 50)  # Task end separator
    
    return {
        "train_loader": train_loader,
        "train_size": len(train_dataset),
        "batch_size": batch_size
    }


# 테스트 데이터를 전처리합니다
def preprocess_test_data(
    test_data: Dict[str, Any],
    parameters: Dict[str, Any]
) -> Dict[str, Any]:
    """Preprocess test data for inference.
    
    Args:
        test_data: Test data dictionary
        parameters: Processing parameters
        
    Returns:
        Preprocessed test data
    """
    
    logger.info("=" * 50)  # Task start separator
    logger.info("Preprocessing labeled test data for evaluation")
    logger.info("=" * 50)
    
    image_size = parameters["data_processing"]["image_size"]
    batch_size = parameters["data_processing"]["batch_size"]
    num_workers = parameters["data_processing"]["num_workers"]
    
    # Log key preprocessing parameters
    logger.info(f"Image size: {image_size[0]}x{image_size[1]}")
    logger.info(f"Batch size: {batch_size:,}")
    logger.info(f"Number of workers: {num_workers}")
    
    # Define transforms (no augmentation for test)
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    image_paths = test_data["image_paths"]
    labels = test_data["labels"]  # Now we have real labels for evaluation
    
    test_dataset = ImageDataset(image_paths, labels, transform=transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    logger.info(f"Test samples: {len(test_dataset):,}")
    logger.info(f"Normal: {labels.count(0):,}, Defective: {labels.count(1):,}")
    logger.info("=" * 50)  # Task end separator
    
    return {
        "test_loader": test_loader,
        "image_paths": image_paths,
        "true_labels": labels,  # Store true labels for evaluation
        "test_size": len(test_dataset)
    }
