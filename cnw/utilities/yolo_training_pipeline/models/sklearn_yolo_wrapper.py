#!/usr/bin/env python3
import os
import sys
import numpy as np
import shutil
import tempfile
from typing import List 
from pathlib import Path
from sklearn.base import BaseEstimator, ClassifierMixin
from ultralytics import YOLO
import math

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from utils.logging_utils import setup_logger
from utils.reproducibility import set_seed


class YOLOSklearnWrapper(BaseEstimator, ClassifierMixin):
    """
    YOLO wrapper for nested CV that handles dataset splits via temporary list files.
    """
    
    def __init__(self, model_name: str = "yolov8n.pt", epochs: int = 100, 
                 imgsz: int = 640, batch: int = 16, device: str = "0", 
                 random_state: int = 42, patience: int = 50, 
                 data_path: str = None, workers: int = 2, **kwargs):
        
        # Store parameters
        self.model_name = model_name
        self.epochs = epochs
        self.imgsz = imgsz
        self.batch = batch
        self.device = device
        self.random_state = random_state
        self.patience = patience
        self.data_path = data_path
        self.workers = workers
        
        self.hyperparams = {k: v for k, v in kwargs.items() if k not in ['pin_memory', 'persistent_workers']}
        
        # Initialize
        self.logger = setup_logger(f"yolo.{Path(model_name).stem}")
        self.model = YOLO(model_name)
        self.is_fitted = False
        self.temp_dir = None
        
        # Required by sklearn, but we get the real number of classes from data.yaml
        self.classes_ = None 
        
        self.logger.info(f"  YOLO {model_name} initialized")
        self.logger.info(f"    Device: {device}")
        self.logger.info(f"   📁 Dataset: {data_path}")

    def fit(self, X_train: List[str], y_train_matrix: np.ndarray, 
            X_val: List[str] = None, y_val_matrix: np.ndarray = None):
        """Train YOLO model on pre-split training and validation data."""
        try:
            set_seed(self.random_state, deterministic=True)
            
            # Create a temporary directory for data.yaml and image lists
            self._cleanup_temp_dir()
            self.temp_dir = tempfile.mkdtemp(prefix="yolo_train_")
            
            # Create the data.yaml file pointing to original image/label directories
            self._create_data_yaml(X_train, X_val)
            
            self.logger.info(f"🚀 Training on {len(X_train)} images. Validating on {len(X_val) if X_val else 0} images.")
            
            train_args = {
                'data': os.path.join(self.temp_dir, 'data.yaml'),
                'epochs': self.epochs,
                'imgsz': self.imgsz,
                'batch': self.batch,
                'device': self.device,
                'patience': self.patience,
                'deterministic': True,
                'cache': 'disk',
                'seed': self.random_state,
                'workers': self.workers,
                'verbose': False,
                'plots': True,
                'save': True,
                'val': True if X_val else False,
                **self.hyperparams
            }
            
            # Train model
            results = self.model.train(**train_args)
            self.is_fitted = True
            
            self.logger.info("✅ Training completed successfully")
            return self
            
        except Exception as e:
            self.logger.error(f"❌ Training failed: {e}")
            raise
        finally:
            self._cleanup_temp_dir()
    
    def score(self, X_test: List[str], y_test: np.ndarray) -> float:
        """Calculate validation score (mAP50-95)"""
        val_temp_dir = None
        try:
            # Create validation data.yaml and image lists
            val_temp_dir = tempfile.mkdtemp(prefix="yolo_val_")
            self.logger.info(f"📋 Created data.yaml and image lists in {val_temp_dir}")
            
            # Robust validation data creation
            val_data_yaml = self._create_validation_data_yaml(X_test, y_test, val_temp_dir)
            
            if val_data_yaml is None or not os.path.exists(val_data_yaml):
                self.logger.error(f"❌ Failed to create validation data.yaml")
                return 0.0
            
            # Run validation
            self.logger.info(f"📊 Validating on {len(X_test)} test images.")
            results = self.model.val(
                data=val_data_yaml,
                device=self.device,
                verbose=False,
                plots=False,
                save=False
            )
            
            # Extract mAP50-95 properly
            map50_95 = 0.0
            
            # Multiple extraction methods
            try:
                if hasattr(results, 'box') and hasattr(results.box, 'map'):
                    map50_95 = float(results.box.map)
                elif hasattr(results, 'results') and len(results.results) > 3:
                    map50_95 = float(results.results[3])
                elif hasattr(results, 'maps') and len(results.maps) > 0:
                    map50_95 = float(results.maps[0])
                else:
                    # Last resort: parse from string representation
                    result_str = str(results)
                    if 'mAP50-95' in result_str:
                        import re
                        match = re.search(r'mAP50-95.*?(\d+\.?\d*)', result_str)
                        if match:
                            map50_95 = float(match.group(1))
            except Exception as extract_error:
                self.logger.warning(f"Score extraction failed: {extract_error}")
                map50_95 = 0.0
            
            # ✅ CRITICAL: Ensure valid return value
            if map50_95 is None or not isinstance(map50_95, (int, float)) or math.isnan(map50_95):
                self.logger.warning(f"Invalid mAP score: {map50_95}, returning 0.0")
                map50_95 = 0.0
            
            map50_95 = max(0.0, float(map50_95))  # Ensure non-negative float
            self.logger.info(f"📈 Validation mAP50-95: {map50_95:.6f}")
            
            return map50_95
            
        except Exception as e:
            self.logger.error(f"❌ Scoring failed: {e}")
            import traceback
            self.logger.debug(f"Full traceback: {traceback.format_exc()}")
            return 0.0
        finally:
            # Clean up temp directory
            if val_temp_dir and os.path.exists(val_temp_dir):
                try:
                    shutil.rmtree(val_temp_dir, ignore_errors=True)
                except:
                    pass

    def _create_validation_data_yaml(self, X_test: List[str], y_test: np.ndarray, temp_dir: str) -> str:
        """Create validation data.yaml file"""
        try:
            # Create val.txt with test images
            val_txt_path = os.path.join(temp_dir, 'val.txt')
            with open(val_txt_path, 'w') as f:
                for img_path in X_test:
                    if os.path.exists(img_path):
                        f.write(f"{img_path}\n")
            
            
            train_txt_path = os.path.join(temp_dir, 'train.txt')
            with open(train_txt_path, 'w') as f:
                f.write("")  # Empty file for validation-only
            
            # Create data.yaml
            data_yaml_path = os.path.join(temp_dir, 'data.yaml')
            
            # Get class names from dataset
            class_names = []
            if hasattr(self, 'data_path') and self.data_path:
                # Try to read classes from original data.yaml
                original_yaml = os.path.join(self.data_path, 'data.yaml')
                if os.path.exists(original_yaml):
                    import yaml
                    with open(original_yaml, 'r') as f:
                        original_data = yaml.safe_load(f)
                        class_names = original_data.get('names', [])
            
            # Fallback: generate class names
            if not class_names:
                num_classes = y_test.shape[1] if len(y_test.shape) > 1 else int(y_test.max()) + 1
                class_names = [f'class_{i}' for i in range(num_classes)]
            
            # Write data.yaml
            with open(data_yaml_path, 'w') as f:
                f.write(f"train: {train_txt_path}\n")
                f.write(f"val: {val_txt_path}\n")
                f.write(f"nc: {len(class_names)}\n")
                f.write(f"names: {class_names}\n")
            
            return data_yaml_path
            
        except Exception as e:
            self.logger.error(f"❌ Failed to create validation data.yaml: {e}")
            return None
    
    def _create_data_yaml(self, X_train: List[str], X_val: List[str]):
        """Create data.yaml and text files pointing to image paths."""
        
        # Original dataset info
        original_yaml_path = os.path.join(self.data_path, 'data.yaml')
        if not os.path.exists(original_yaml_path):
            raise FileNotFoundError(f"Missing data.yaml at {original_yaml_path}")
        
        # Read original data.yaml
        import yaml
        with open(original_yaml_path, 'r') as f:
            original_data = yaml.safe_load(f)
        
        train_list_path = os.path.join(self.temp_dir, 'train.txt')
        val_list_path = os.path.join(self.temp_dir, 'val.txt')

        # Write image paths to respective files
        if X_train:
            with open(train_list_path, 'w') as f:
                f.write('\n'.join(X_train))
        
        if X_val:
            with open(val_list_path, 'w') as f:
                f.write('\n'.join(X_val))

        # Create new data.yaml
        yaml_content = {
            'path': self.data_path,  # Point to the original dataset root
            'train': train_list_path,
            'val': val_list_path,
            'nc': original_data['nc'],
            'names': original_data['names']
        }
        
        with open(os.path.join(self.temp_dir, 'data.yaml'), 'w') as f:
            yaml.safe_dump(yaml_content, f)

        self.logger.info(f"   📋 Created data.yaml and image lists in {self.temp_dir}")
        
    def _cleanup_temp_dir(self):
        """Clean up temporary directory."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
            except Exception as e:
                self.logger.warning(f"Failed to remove temp dir {self.temp_dir}: {e}")
        self.temp_dir = None
    
    def __del__(self):
        """Cleanup on destruction."""
        self._cleanup_temp_dir()