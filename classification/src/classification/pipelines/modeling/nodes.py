"""Modeling nodes for CNN classification."""
import logging
import time
from typing import Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


logger = logging.getLogger(__name__)


class CNNModel(nn.Module):
    """CNN model for binary classification."""
    
    def __init__(self, input_channels: int, num_classes: int, conv_layers: list, fc_layers: list, 
                 dropout_rate: float = 0.5, use_batch_norm: bool = True, pooling_config: dict = None):
        super(CNNModel, self).__init__()
        
        # Set default pooling configuration if not provided
        if pooling_config is None:
            pooling_config = {
                "max_pool": {"kernel_size": 2, "stride": 2, "padding": 0},
                "adaptive_pool": {"output_size": [7, 7]}
            }
        
        self.features = nn.ModuleList()
        in_channels = input_channels
        max_pool_config = pooling_config["max_pool"]
        
        # Convolutional layers
        for layer_config in conv_layers:
            out_channels = layer_config['out_channels']
            kernel_size = layer_config['kernel_size']
            stride = layer_config.get('stride', 1)
            padding = layer_config.get('padding', 0)
            
            # Build conv block based on batch_norm setting
            conv_block_layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)]
            
            if use_batch_norm:
                conv_block_layers.append(nn.BatchNorm2d(out_channels))
            
            conv_block_layers.extend([
                nn.ReLU(inplace=True),
                nn.MaxPool2d(
                    kernel_size=max_pool_config["kernel_size"],
                    stride=max_pool_config["stride"],
                    padding=max_pool_config["padding"]
                )
            ])
            
            conv_block = nn.Sequential(*conv_block_layers)
            self.features.append(conv_block)
            in_channels = out_channels
        
        # Adaptive pooling to handle different input sizes
        adaptive_size = pooling_config["adaptive_pool"]["output_size"]
        self.adaptive_pool = nn.AdaptiveAvgPool2d(tuple(adaptive_size))
        
        # Calculate the size after convolution and pooling
        feature_size = in_channels * adaptive_size[0] * adaptive_size[1]
        
        # Fully connected layers
        classifier_layers = []
        for fc_size in fc_layers:
            classifier_layers.extend([
                nn.Linear(feature_size, fc_size),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate)
            ])
            feature_size = fc_size
        
        # Final classification layer
        classifier_layers.append(nn.Linear(feature_size, num_classes))
        
        self.classifier = nn.Sequential(*classifier_layers)
    
    def forward(self, x):
        # Feature extraction
        for feature_layer in self.features:
            x = feature_layer(x)
        
        # Adaptive pooling
        x = self.adaptive_pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Classification
        x = self.classifier(x)
        
        return x


# CNN 모델을 생성하고 초기화합니다
def create_model(parameters: Dict[str, Any]) -> nn.Module:
    """Create and initialize CNN model.
    
    Args:
        parameters: Model configuration parameters
        
    Returns:
        Initialized CNN model
    """
    logger.info("Creating CNN model")
    
    model_params = parameters["model"]
    
    model = CNNModel(
        input_channels=model_params["input_channels"],
        num_classes=model_params["num_classes"],
        conv_layers=model_params["conv_layers"],
        fc_layers=model_params["fc_layers"],
        dropout_rate=model_params.get("dropout_rate", 0.5),
        use_batch_norm=model_params.get("batch_norm", True),
        pooling_config=model_params.get("pooling", None)
    )
    
    # Initialize weights based on method
    init_method = model_params.get("init_method", "kaiming")
    init_gain = model_params.get("init_gain", 1.0)
    
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            if init_method == "kaiming":
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif init_method == "xavier":
                nn.init.xavier_normal_(m.weight, gain=init_gain)
            elif init_method == "normal":
                nn.init.normal_(m.weight, 0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            if init_method == "kaiming":
                nn.init.kaiming_normal_(m.weight)
            elif init_method == "xavier":
                nn.init.xavier_normal_(m.weight, gain=init_gain)
            else:
                nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)
    
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Log model configuration details
    logger.info(f"Model configuration:")
    logger.info(f"  - Dropout rate: {model_params.get('dropout_rate', 0.5)}")
    logger.info(f"  - Batch normalization: {model_params.get('batch_norm', True)}")
    logger.info(f"  - Weight initialization: {init_method}")
    
    # Log pooling configuration
    pooling_config = model_params.get("pooling", {})
    if pooling_config:
        max_pool = pooling_config.get("max_pool", {})
        adaptive_pool = pooling_config.get("adaptive_pool", {})
        logger.info(f"  - MaxPool: kernel={max_pool.get('kernel_size', 2)}, stride={max_pool.get('stride', 2)}")
        logger.info(f"  - AdaptivePool: output_size={adaptive_pool.get('output_size', [7, 7])}")
    
    return model


# 모델을 훈련시키고 성능을 평가합니다
def train_model(
    preprocessed_train_data: Dict[str, Any],
    parameters: Dict[str, Any]
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Train CNN model on full dataset (no validation).
    
    Args:
        preprocessed_train_data: Preprocessed training data
        parameters: Training parameters
        
    Returns:
        Tuple of trained model and training metrics
    """
    logger.info("=" * 50)  # Task start separator
    logger.info("Starting model training on full dataset")
    logger.info("=" * 50)
    
    # Get parameters
    model_params = parameters["model"]
    training_params = parameters["training"]
    
    # Set device
    device = torch.device(model_params["device"] if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Log model architecture
    logger.info(f"Model architecture:")
    logger.info(f"  - Input channels: {model_params['input_channels']}")
    logger.info(f"  - Number of classes: {model_params['num_classes']}")
    logger.info(f"  - CNN layers: {len(model_params['conv_layers'])}")
    for i, layer in enumerate(model_params['conv_layers']):
        logger.info(f"    Conv{i+1}: {layer['out_channels']} channels, {layer['kernel_size']}x{layer['kernel_size']} kernel")
    logger.info(f"  - FC layers: {model_params['fc_layers']}")
    
    # Log training configuration
    logger.info(f"Training configuration:")
    logger.info(f"  - Learning rate: {model_params['learning_rate']}")
    logger.info(f"  - Epochs: {model_params['epochs']}")
    logger.info(f"  - Optimizer: {training_params['optimizer']}")
    logger.info(f"  - Weight decay: {model_params.get('weight_decay', 0.0)}")
    
    # Create model
    model = create_model(parameters)
    model = model.to(device)
    
    # Get data loader
    train_loader = preprocessed_train_data["train_loader"]
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    
    # Create optimizer with weight decay
    weight_decay = model_params.get("weight_decay", 0.0)
    optimizer_name = training_params["optimizer"]
    
    if optimizer_name == "adam":
        adam_params = training_params.get("adam", {})
        optimizer = optim.Adam(
            model.parameters(), 
            lr=model_params["learning_rate"],
            weight_decay=weight_decay,
            betas=(adam_params.get("beta1", 0.9), adam_params.get("beta2", 0.999)),
            eps=adam_params.get("eps", 1e-8)
        )
    elif optimizer_name == "sgd":
        sgd_params = training_params.get("sgd", {})
        optimizer = optim.SGD(
            model.parameters(), 
            lr=model_params["learning_rate"],
            weight_decay=weight_decay,
            momentum=sgd_params.get("momentum", 0.9),
            nesterov=sgd_params.get("nesterov", False)
        )
    elif optimizer_name == "adamw":
        adam_params = training_params.get("adam", {})
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=model_params["learning_rate"],
            weight_decay=weight_decay,
            betas=(adam_params.get("beta1", 0.9), adam_params.get("beta2", 0.999)),
            eps=adam_params.get("eps", 1e-8)
        )
    else:
        optimizer = optim.Adam(model.parameters(), lr=model_params["learning_rate"], weight_decay=weight_decay)
    
    # Learning rate scheduler
    if training_params["scheduler"] == "step":
        scheduler = lr_scheduler.StepLR(
            optimizer, 
            step_size=training_params["step_size"], 
            gamma=training_params["gamma"]
        )
    else:
        scheduler = None
    
    # Training loop
    num_epochs = model_params["epochs"]
    train_losses = []
    train_accs = []
    
    # Calculate dynamic logging interval (aim for 5 logs per epoch)
    total_batches = len(train_loader)
    log_interval = max(1, total_batches // 5)
    
    logger.info(f"Training for {num_epochs} epochs")
    logger.info(f"Total batches per epoch: {total_batches:,}, logging every {log_interval:,} batches")
    logger.info("=" * 50)
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Training phase
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            
            # Dynamic logging based on total batch count
            if batch_idx % log_interval == 0:
                progress_pct = (batch_idx + 1) / total_batches * 100
                logger.info(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1:,}/{total_batches:,} ({progress_pct:.1f}%), Loss: {loss.item():.4f}")
        
        train_loss = running_loss / len(train_loader)
        train_acc = correct_predictions / total_predictions
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Update learning rate
        if scheduler:
            scheduler.step()
        
        epoch_time = time.time() - start_time
        logger.info(f"Epoch {epoch+1}/{num_epochs} completed in {epoch_time:.2f}s")
        logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
    
    # Training metrics
    final_train_acc = train_accs[-1] if train_accs else 0.0
    training_metrics = {
        "train_losses": train_losses,
        "train_accuracies": train_accs,
        "final_train_accuracy": final_train_acc,
        "total_epochs": num_epochs,
        "model_parameters": sum(p.numel() for p in model.parameters())
    }
    
    logger.info(f"Training completed. Final training accuracy: {final_train_acc:.4f}")
    logger.info("=" * 50)  # Task end separator
    
    return model, training_metrics



