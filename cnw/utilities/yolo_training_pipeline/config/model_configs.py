"""
Model configurations for YOLO variants.

This file acts as a single source of truth for all supported YOLO models.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class ModelConfig:
    """Configuration for a specific YOLO model variant."""
    name: str
    file_path: str
    input_size: int
    batch_size: int
    description: str
    
    @property
    def model_path(self) -> str:
        """Alias for file_path for consistency"""
        return self.file_path
    
    def get_adjusted_batch_size(self, base_config) -> int:
        """Get batch size adjusted for available GPUs"""
        try:
            gpu_count = max(1, base_config.get_gpu_count())
            # Scale batch size with number of GPUs, but cap it
            return min(self.batch_size * gpu_count, self.batch_size * 4)
        except:
            # Fallback if get_gpu_count() doesn't exist
            return self.batch_size

# Model configurations dictionary
MODEL_CONFIGS = {
    # YOLOv5 variants
    "yolov5n": ModelConfig(
        name="yolov5n",
        file_path="yolov5n.pt",
        input_size=640,
        batch_size=32,
        description="YOLOv5 Nano - Fastest, smallest model"
    ),
    "yolov5s": ModelConfig(
        name="yolov5s",
        file_path="yolov5s.pt", 
        input_size=640,
        batch_size=16,
        description="YOLOv5 Small - Good speed/accuracy balance"
    ),
    "yolov5mu": ModelConfig(
        name="yolov5mu",
        file_path="yolov5mu.pt",
        input_size=640,
        batch_size=8,
        description="YOLOv5 Medium - Higher accuracy"
    ),
    "yolov5lu": ModelConfig(
        name="yolov5lu",
        file_path="yolov5lu.pt",
        input_size=640,
        batch_size=4,
        description="YOLOv5 Large - High accuracy"
    ),
    "yolov5x": ModelConfig(
        name="yolov5x",
        file_path="yolov5x.pt",
        input_size=640,
        batch_size=2,
        description="YOLOv5 Extra Large - Highest accuracy"
    ),
    "yolov5su": ModelConfig(
        name="yolov5su",
        file_path="yolov5su.pt",
        input_size=640,
        batch_size=8,
        description="YOLOv5 Small-U - Ultralytics optimized"
    ),
    
    # YOLOv8 variants
    "yolov8n": ModelConfig(
        name="yolov8n",
        file_path="yolov8n.pt",
        input_size=640,
        batch_size=32,
        description="YOLOv8 Nano - Latest architecture, fast"
    ),
    "yolov8s": ModelConfig(
        name="yolov8s",
        file_path="yolov8s.pt",
        input_size=640,
        batch_size=16,
        description="YOLOv8 Small - Latest architecture, balanced"
    ),
    "yolov8m": ModelConfig(
        name="yolov8m",
        file_path="yolov8m.pt",
        input_size=640,
        batch_size=8,
        description="YOLOv8 Medium - Latest architecture, accurate"
    ),
    "yolov8l": ModelConfig(
        name="yolov8l",
        file_path="yolov8l.pt",
        input_size=640,
        batch_size=4,
        description="YOLOv8 Large - Latest architecture, high accuracy"
    ),
    "yolov8x": ModelConfig(
        name="yolov8x",
        file_path="yolov8x.pt",
        input_size=640,
        batch_size=2,
        description="YOLOv8 Extra Large - Latest architecture, highest accuracy"
    ),

    # YOLOv11 variants
    "yolo11s": ModelConfig(
        name="yolo11s",
        file_path="yolo11s.pt",
        input_size=640,
        batch_size=16,
        description="YOLOv11 Small - Optimized for speed and balance"
    ),
    "yolo11m": ModelConfig(
        name="yolo11m",
        file_path="yolo11m.pt",
        input_size=640,
        batch_size=8,
        description="YOLOv11 Medium - Balanced accuracy and performance"
    ),
    "yolo11l": ModelConfig(
        name="yolo11l",
        file_path="yolo11l.pt",
        input_size=640,
        batch_size=4,
        description="YOLOv11 Large - High accuracy for complex tasks"
    ),

    # YOLOv12 variants
    "yolo12s": ModelConfig(
        name="yolo12s",
        file_path="yolo12s.pt",
        input_size=640,
        batch_size=16,
        description="YOLOv12 Small - Next-gen architecture, fast and light"
    ),
    "yolo12m": ModelConfig(
        name="yolo12m",
        file_path="yolo12m.pt",
        input_size=640,
        batch_size=8,
        description="YOLOv12 Medium - Balanced next-gen model"
    ),
    "yolo12l": ModelConfig(
        name="yolo12l",
        file_path="yolo12l.pt",
        input_size=640,
        batch_size=4,
        description="YOLOv12 Large - Next-gen model for high-accuracy applications"
    ),
}

# List of available model names
AVAILABLE_MODELS = list(MODEL_CONFIGS.keys())

def get_model_config(model_name: str) -> ModelConfig:
    """
    Get model configuration by name.
    
    Args:
        model_name: Name of the model (e.g., 'yolov5s', 'yolov8n')
        
    Returns:
        ModelConfig instance
        
    Raises:
        ValueError: If model_name is not supported
    """
    if model_name not in MODEL_CONFIGS:
        available = ", ".join(AVAILABLE_MODELS)
        raise ValueError(f"Model '{model_name}' not supported. Available models: {available}")
    
    return MODEL_CONFIGS[model_name]

def list_available_models() -> List[str]:
    """Return list of all available model names."""
    return AVAILABLE_MODELS.copy()

def get_model_info() -> Dict[str, str]:
    """Return dictionary mapping model names to descriptions."""
    return {name: config.description for name, config in MODEL_CONFIGS.items()}
