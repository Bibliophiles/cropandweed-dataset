from dataclasses import dataclass, field
from typing import Dict, Optional, Any
import argparse


@dataclass
class BaseConfig:
    """
    A passive data container for all base configuration parameters.
    This class should not contain any setup or validation logic.
    """
    
    # Data paths
    data_path: str
    output_dir: str = "./outputs"
    
    # Cross-validation settings
    n_folds: int = 5
    n_outer_folds: int = 5
    random_state: int = 42
    
    # Optuna settings
    n_trials: int = 100
    study_name: Optional[str] = None
    storage_url: Optional[str] = None
    
    # Training settings
    epochs: int = 100
    patience: int = 20
    device: str = "auto"
    workers: int = 2
    
    # Reproducibility
    seed: int = 42
    deterministic: bool = True
    
    # Logging
    log_level: str = "INFO"
    wandb_project: Optional[str] = None
    
    # Data Augmentation Settings
    enable_augmentation: bool = True
    augmentation_params: Optional[Dict[str, Any]] = None

    # GPU Configuration
    gpu_ids: Optional[str] = "auto"  

    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser):
        """Adds all BaseConfig arguments to an argparse parser."""
        parser.add_argument('--data_path', type=str, required=True,
                            help='Path to dataset directory')
        parser.add_argument('--output_dir', type=str, default='./outputs',
                            help='Directory to save all outputs')
        parser.add_argument('--n_outer_folds', type=int, default=5,
                            help='Number of outer CV folds')
        parser.add_argument('--n_folds', type=int, default=5,
                            help='Number of inner CV folds')
        parser.add_argument('--random_state', type=int, default=42,
                            help='Random seed for reproducibility')
        parser.add_argument('--n_trials', type=int, default=100,
                            help='Number of Optuna trials per fold')
        parser.add_argument('--epochs', type=int, default=100,
                            help='Number of training epochs')
        parser.add_argument('--patience', type=int, default=20,
                            help='Early stopping patience')
        parser.add_argument('--device', type=str, default="auto",
                            help='Device to use for training (e.g., "0", "1", "cpu", "auto")')
        parser.add_argument('--workers', type=int, default=4,
                            help='Number of data loading workers')
        parser.add_argument('--deterministic', action='store_true',
                            help='Enable deterministic behavior for reproducibility')
        parser.add_argument('--log_level', type=str, default='INFO',
                            choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                            help='Logging level')
        parser.add_argument('--wandb_project', type=str, default=None,
                            help='Weights & Biases project name')
        parser.add_argument('--gpu_ids', type=str, default="auto",
                            help='GPU IDs to use (e.g., "0", "0,1", "cpu", "auto")')

    @classmethod
    def from_argparse(cls, args: argparse.Namespace):
        """Creates a BaseConfig instance from a parsed argparse namespace."""
        return cls(
            data_path=args.data_path,
            output_dir=args.output_dir,
            n_folds=args.n_folds,
            n_outer_folds=args.n_outer_folds,
            random_state=args.random_state,
            n_trials=args.n_trials,
            epochs=args.epochs,
            patience=args.patience,
            device=args.device,
            workers=args.workers,
            seed=args.random_state, 
            deterministic=args.deterministic,
            log_level=args.log_level,
            wandb_project=args.wandb_project,
            gpu_ids=args.gpu_ids, 
        )

    def get_device_string(self) -> str:
        """Get device string for YOLO training"""
        from utils.utils import parse_gpu_ids, format_device_string
        gpu_list = parse_gpu_ids(self.gpu_ids)
        return format_device_string(gpu_list)
    
    def get_gpu_count(self) -> int:
        """Get number of GPUs to use"""
        from utils.utils import parse_gpu_ids
        return len(parse_gpu_ids(self.gpu_ids))

@dataclass
class ModelConfig:
    """Configuration for a specific YOLO model"""
    model_name: str
    model_path: str
    input_size: int = 640
    batch_size: int = 16
    description: str = ""

    def __post_init__(self):
        # Basic validation
        if not self.model_name:
            raise ValueError("model_name cannot be empty")
        if not self.model_path:
            raise ValueError("model_path cannot be empty")
        if self.input_size <= 0:
            raise ValueError("input_size must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")