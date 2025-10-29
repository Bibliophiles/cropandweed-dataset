#!/usr/bin/env python3

import random
import numpy as np
import torch
import os
import sys
import platform

def set_seed(seed: int, deterministic: bool = True):
    """
    Set random seeds for reproducibility across all libraries.
    
    Args:
        seed: Random seed value.
        deterministic: Whether to use deterministic operations.
                       This may reduce performance.
    """
    
    # 1. Python, NumPy, and system-level seeds
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # 2. PyTorch seed settings
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # For multi-GPU
    
    # 3. Deterministic behavior for PyTorch operations
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # This flag is necessary for some PyTorch versions to make
        # specific CUDA operations (like `torch.addcmul`) deterministic.
        # It MUST be set BEFORE any CUDA calls.
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

        # This ensures all deterministic algorithms are used.
        # Note: Not all algorithms have a deterministic implementation.
        if hasattr(torch, 'use_deterministic_algorithms'):
            torch.use_deterministic_algorithms(True)

    print(f"🎲 Random seed set to: {seed} (deterministic: {deterministic})")

def get_device_info() -> dict:
    """
    Get information about available compute devices.
    
    Returns:
        Dictionary with device information.
    """
    
    device_info = {
        'cpu_available': True,
        'cuda_available': torch.cuda.is_available(),
        'cuda_device_count': 0,
        'cuda_devices': []
    }
    
    if torch.cuda.is_available():
        device_info['cuda_device_count'] = torch.cuda.device_count()
        
        for i in range(torch.cuda.device_count()):
            try:
                props = torch.cuda.get_device_properties(i)
                device_info['cuda_devices'].append({
                    'id': i,
                    'name': props.name,
                    'total_memory_gb': props.total_memory / (1024**3),
                    'compute_capability': f"{props.major}.{props.minor}"
                })
            except Exception as e:
                device_info['cuda_devices'].append({
                    'id': i,
                    'error': str(e)
                })
    
    return device_info

def get_environment_info() -> dict:
    """
    Get information about the current environment.
    
    Returns:
        Dictionary with environment information.
    """
    
    env_info = {
        'platform': platform.platform(),
        'python_version': sys.version,
        'pytorch_version': torch.__version__,
    }
    
    # Add key library versions
    libs_to_check = {
        'ultralytics': 'ultralytics',
        'optuna': 'optuna',
        'numpy': 'numpy',
        'sklearn': 'sklearn',
        'skmultilearn': 'skmultilearn',
        'pandas': 'pandas',
        'scipy': 'scipy'
    }
    
    for lib_name, import_name in libs_to_check.items():
        try:
            lib = __import__(import_name)
            env_info[f'{lib_name}_version'] = lib.__version__
        except (ImportError, AttributeError):
            env_info[f'{lib_name}_version'] = 'Not installed or version not available'
            
    return env_info

def check_environment() -> bool:
    """
    Check if the environment is properly set up.
    
    Returns:
        True if environment is ready, False otherwise.
    """
    
    print("🔍 Checking environment...")
    
    required_packages = {
        'torch': 'torch',
        'ultralytics': 'ultralytics',
        'optuna': 'optuna',
        'numpy': 'numpy',
        'sklearn': 'scikit-learn',
        'skmultilearn': 'scikit-multilearn',
        'pandas': 'pandas',
        'scipy': 'scipy'
    }
    
    missing_packages = []
    
    for import_name, package_name in required_packages.items():
        try:
            __import__(import_name)
            print(f"✅ {package_name} ({import_name}) is available")
        except ImportError:
            print(f"❌ {package_name} ({import_name}) is missing")
            missing_packages.append(package_name)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {missing_packages}")
        print("Install them with: pip install " + " ".join(missing_packages))
        return False
    
    device_info = get_device_info()
    if device_info['cuda_available']:
        print(f"\n✅ CUDA available with {device_info['cuda_device_count']} GPU(s)")
        for gpu in device_info['cuda_devices']:
            print(f"   GPU {gpu.get('id', 'N/A')}: {gpu.get('name', 'N/A')} ({gpu.get('total_memory_gb', 0):.1f} GB)")
    else:
        print("\n⚠️  CUDA not available, will use CPU")
    
    print("\n✅ Environment check complete")
    return True