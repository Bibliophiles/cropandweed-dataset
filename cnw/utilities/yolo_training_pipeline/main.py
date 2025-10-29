import argparse
import logging
import os
import sys
import multiprocessing as mp
from pathlib import Path
import gc
import torch
from datetime import datetime
import json

# Add parent directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
torch.multiprocessing.set_sharing_strategy('file_system')

from config.base_config import BaseConfig
from config.model_configs import get_model_config, AVAILABLE_MODELS
from optimization.optuna_nested_cv import OptunaNestedCV
from utils.reproducibility import set_seed, check_environment
from utils.device_utils import DeviceManager
from utils.logging_utils import LoggingManager
from utils.visualization import PipelineVisualizer
from utils.gpu_monitoring import GPUMonitorContext  

def parse_arguments():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="YOLO Optuna Nested Cross-Validation")
    
    # Arguments from BaseConfig
    BaseConfig.add_arguments(parser)

    # Model-specific arguments
    parser.add_argument('--model', type=str, required=True, 
                        choices=AVAILABLE_MODELS,
                        help='YOLO model to train')
    
    # Custom flags
    parser.add_argument('--nested_cv', action='store_true',
                        help='Enable nested cross-validation')

    return parser.parse_args()

def validate_dataset(data_path: str, logger: logging.Logger):
    """
    Validate dataset structure and content before training.
    """
    data_path = Path(data_path)
    
    if not data_path.is_dir():
        raise FileNotFoundError(f"Dataset path does not exist or is not a directory: {data_path}")

    # Check required files and directories
    data_yaml = data_path / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found in {data_path}")
    
    images_dir = data_path / "images"
    labels_dir = data_path / "labels"
    
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images directory missing: {images_dir}")
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"Labels directory missing: {labels_dir}")
    
    # Count files
    image_files = [f for f in images_dir.rglob('*') if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
    label_files = [f for f in labels_dir.rglob('*.txt')]
    
    logger.info(f"📊 Dataset validation:")
    logger.info(f"   📁 Dataset path: {data_path.absolute()}")
    logger.info(f"   🖼️  Images found: {len(image_files)}")
    logger.info(f"   🏷️  Labels found: {len(label_files)}")
    
    if len(image_files) == 0:
        raise ValueError(f"No images found in {images_dir}")
    if len(label_files) == 0:
        logger.warning("No labels found. This dataset might be used for inference only.")
    
    # Check a sample of image-label pairs for consistency
    valid_pairs = 0
    for img_file in image_files[:10]:
        label_file = labels_dir / f"{img_file.stem}.txt"
        if label_file.exists():
            valid_pairs += 1
    
    if valid_pairs > 0:
        logger.info(f"   ✅ Found {valid_pairs}/10 valid image-label pairs (sample check).")
    else:
        logger.warning("No valid image-label pairs found in the sample check.")
    
    # Validate data.yaml
    try:
        import yaml
        with open(data_yaml, 'r') as f:
            config = yaml.safe_load(f)
        
        required_keys = ['nc', 'names']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required key in data.yaml: {key}")
        
        logger.info(f"   📋 Classes: {config['nc']} ({config['names'][:3]}...)")
        
    except Exception as e:
        raise ValueError(f"Invalid data.yaml: {e}") from e
    
    logger.info(f"   ✅ Dataset validation passed!")


def cleanup_memory():
    """
    Clean up GPU/CPU memory.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_nested_cv_pipeline(model_config, base_config, logger):
    """Runs the nested cross-validation pipeline with hardware monitoring."""
    logger.info("🚀 Starting Optuna Nested Cross-Validation")
    
    # Create hardware monitoring file path using existing structure
    output_dir = Path(base_config.output_dir)
    data_path_name = Path(base_config.data_path).name
    experiment_name = f"{model_config.name}_{data_path_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    hardware_log_path = output_dir / f"{experiment_name}_hardware.json"
    
    # Use existing GPU monitoring with context manager
    with GPUMonitorContext(save_path=str(hardware_log_path)) as monitor:
        nested_cv = OptunaNestedCV(
            model_config=model_config,
            base_config=base_config,
            data_path=base_config.data_path,
            workers=base_config.workers
        )
        
        results = nested_cv.run_nested_cv()
        
        # Get hardware data from existing monitor
        hardware_stats = monitor.get_summary_stats()
        results['hardware_usage'] = monitor.monitoring_data
        results['hardware_summary'] = hardware_stats
    
    # Log hardware summary using existing logger
    if hardware_stats:
        logger.info(f"Hardware Usage Summary:")
        logger.info(f"   Training Duration: {hardware_stats.get('duration_minutes', 0):.1f} minutes")
        
        for gpu_id in monitor.gpu_ids:
            gpu_key = f'gpu_{gpu_id}'
            if gpu_key in hardware_stats:
                gpu_stats = hardware_stats[gpu_key]
                logger.info(f"   GPU {gpu_id}: {gpu_stats['usage_mean']:.1f}% avg, "
                           f"{gpu_stats['memory_mean_gb']:.1f}GB avg memory")
    
    # Generate visualizations using existing system
    try:
        logger.info("📊 Generating visualizations...")
        
        # Use existing visualization with proper logging
        visualizer = PipelineVisualizer(
            output_dir=str(output_dir),
            experiment_name=experiment_name
        )
        
        # Save results to JSON
        results_file = output_dir / f"{experiment_name}_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Generate all plots
        visualizer.generate_all_plots(str(results_file))
        logger.info(f"Visualizations saved to: {visualizer.viz_dir}")
        
    except Exception as e:
        logger.warning(f"Failed to generate visualizations: {e}")
    
    logger.info("Pipeline completed successfully!")
    logger.info(f"Final Performance: {results['mean_test_score']:.4f} ± {results['std_test_score']:.4f}")
    return results

def run_single_model_pipeline(model_config, base_config, logger):
    """
    Runs a single model training and evaluation pipeline (to be implemented).
    """
    logger.info("🚀 Starting single-model training and evaluation (not implemented yet)")
    logger.info("This feature will be implemented in a future update.")
    # Placeholder for single-model training logic
    # You would typically have a separate class or function here for a single train/val/test run
    return {}

def main():
    """Main function with proper configuration handling"""
    mp.set_start_method('spawn', force=True)

    # 1. Parse arguments and create configs
    args = parse_arguments()
    base_config = BaseConfig.from_argparse(args)
    model_config = get_model_config(args.model)
    
    # 2. Setup logging
    logging_manager = LoggingManager(
        output_dir=base_config.output_dir, 
        model_name=model_config.name, 
        log_level=base_config.log_level
    )
    logger = logging_manager.get_logger("main")
    
    logger.info("🚀 YOLO Pipeline Started")
    logger.info(f"    Model: {model_config.name}")
    logger.info(f"    Dataset: {base_config.data_path}")
    logger.info(f"    Device: {base_config.get_device_string()}")
    logger.info(f"    GPU Count: {base_config.get_gpu_count()}")
    
    # 3. Environment setup
    if not check_environment():
        logger.error("❌ Environment check failed")
        sys.exit(1)
    
    set_seed(base_config.seed, base_config.deterministic)
    
    # 4. Device setup
    device_manager = DeviceManager()
    device_manager.print_device_summary()
    
    # 5. Validate dataset
    try:
        validate_dataset(base_config.data_path, logger)
    except Exception as e:
        logger.error(f"❌ Dataset validation failed: {e}")
        sys.exit(1)
    
    # 6. Run training pipeline
    try:
        if args.nested_cv:
            logger.info("🔄 Starting Nested Cross-Validation")
            results = run_nested_cv_pipeline(model_config, base_config, logger)
        else:
            logger.info("🔄 Starting Single Model Training")
            results = run_single_model_pipeline(model_config, base_config, logger)
        
        logger.info("🎉 Pipeline completed successfully!")
        
        # Generate final summary
        logger.info(" Final Summary:")
        if 'mean_test_score' in results:
            logger.info(f"    Performance: {results['mean_test_score']:.4f} ± {results.get('std_test_score', 0):.4f}")
        logger.info(f"   📁 Results saved to: {base_config.output_dir}")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        import traceback
        logger.error(f"Stack trace: {traceback.format_exc()}")
        sys.exit(1)
    finally:
        cleanup_memory()


if __name__ == '__main__':
    main()