#!/usr/bin/env python3

import torch
from typing import List, Union

def parse_gpu_ids(gpu_ids: Union[str, List[str], None]) -> List[int]:
    """
    Parse GPU IDs from various input formats.
    
    Args:
        gpu_ids: Can be:
            - "0,1,2" (comma-separated string)
            - ["0", "1", "2"] (list of strings)
            - "0" (single GPU)
            - "cpu" (use CPU)
            - None (auto-detect)
    
    Returns:
        List of GPU IDs to use
    """
    if gpu_ids is None or gpu_ids == "auto":
        # Auto-detect available GPUs
        if torch.cuda.is_available():
            return list(range(torch.cuda.device_count()))
        else:
            return []
    
    if gpu_ids == "cpu":
        return []
    
    if isinstance(gpu_ids, str):
        if "," in gpu_ids:
            # Parse comma-separated string
            gpu_list = [int(x.strip()) for x in gpu_ids.split(",") if x.strip()]
        else:
            # Single GPU as string
            gpu_list = [int(gpu_ids)]
    elif isinstance(gpu_ids, (list, tuple)):
        # List or tuple of GPU IDs
        gpu_list = [int(x) for x in gpu_ids]
    else:
        raise ValueError(f"Invalid gpu_ids format: {gpu_ids}")
    
    # Validate GPU IDs
    if torch.cuda.is_available():
        available_gpus = list(range(torch.cuda.device_count()))
        invalid_gpus = [g for g in gpu_list if g not in available_gpus]
        if invalid_gpus:
            raise ValueError(f"Invalid GPU IDs {invalid_gpus}. Available: {available_gpus}")
    elif gpu_list:
        raise ValueError("CUDA not available but GPU IDs specified")
    
    return gpu_list

def format_device_string(gpu_ids: List[int]) -> str:
    """Convert GPU ID list to device string for YOLO"""
    if not gpu_ids:
        return "cpu"
    elif len(gpu_ids) == 1:
        return str(gpu_ids[0])
    else:
        return ",".join(map(str, gpu_ids))