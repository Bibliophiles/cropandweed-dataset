# utils/device_utils.py
import torch
from typing import Dict, Union, Any

class DeviceManager:
    """
    Manages GPU-related utilities for model training.
    """

    @staticmethod
    def setup_device(device_id: Union[int, str] = "auto") -> torch.device:
        """
        Sets up the optimal device for training (GPU or CPU).
        
        Args:
            device_id: The ID of the GPU to use (e.g., '0', '1') or 'cpu'. Defaults to "auto".
                       If 'auto', it selects the first available GPU, or the CPU if none are found.
        
        Returns:
            The torch.device object to use for training.
        """
        if isinstance(device_id, int):
            device_id = str(device_id)

        if torch.cuda.is_available() and device_id != 'cpu' and device_id != 'auto':
            try:
                # Validate the specified device ID
                if int(device_id) >= torch.cuda.device_count() or int(device_id) < 0:
                    print(f"⚠️  Specified device {device_id} is not available. Falling back to CPU.")
                    return torch.device('cpu')
                
                device = torch.device(f"cuda:{device_id}")
                print(f"✅ Using specified device: {device}")
                return device
            except (ValueError, IndexError):
                print(f"⚠️  Invalid device ID '{device_id}'. Falling back to CPU.")
                return torch.device('cpu')
        
        if torch.cuda.is_available() and device_id in ['auto', None]:
            device = torch.device("cuda:0")
            print(f"🚀 Found {torch.cuda.device_count()} GPU(s). Using default device: {device}")
            return device

        print("⚠️  CUDA not available, using CPU.")
        return torch.device('cpu')

    @staticmethod
    def get_device_info() -> Dict[str, Any]:
        """
        Retrieves detailed GPU information.
        
        Returns:
            A dictionary with GPU details.
        """
        if not torch.cuda.is_available():
            return {"available": False, "count": 0, "devices": []}
        
        gpu_count = torch.cuda.device_count()
        devices = []
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            devices.append({
                "id": i,
                "name": props.name,
                "total_memory_gb": props.total_memory / (1024**3),
                "compute_capability": f"{props.major}.{props.minor}"
            })
        return {"available": True, "count": gpu_count, "devices": devices}

    @staticmethod
    def print_device_summary():
        """
        Prints a summary of available devices.
        """
        device_info = DeviceManager.get_device_info()
        if not device_info["available"]:
            print("❌ No CUDA GPUs available.")
            return
        
        print(f"🚀 Found {device_info['count']} CUDA GPU(s):")
        for device in device_info["devices"]:
            print(f"   GPU {device['id']}: {device['name']}")
            print(f"      Memory: {device['total_memory_gb']:.1f} GB")
            print(f"      Compute Capability: {device['compute_capability']}")
            
            # Print current usage if available (requires torch.cuda)
            try:
                allocated = torch.cuda.memory_allocated(device['id']) / (1024**3)
                cached = torch.cuda.memory_reserved(device['id']) / (1024**3)
                print(f"      Current usage: {allocated:.1f} GB allocated, {cached:.1f} GB reserved")
            except Exception:
                pass
            print("-" * 50)

    @staticmethod
    def clear_gpu_memory():
        """
        Clears the GPU memory cache to free up memory.
        """
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                print("🧹 GPU memory cache cleared.")
            except Exception as e:
                print(f"⚠️  Failed to clear GPU memory cache: {e}")