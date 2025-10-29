# utils/gpu_monitoring.py
import subprocess
import time
import threading
import logging
import psutil
import json
from typing import List, Dict, Any
from pathlib import Path
from .logging_utils import setup_logger

def get_gpu_utilization(gpu_id: int) -> float:
    """
    Retrieves the GPU utilization percentage using nvidia-smi.
    
    Args:
        gpu_id: The ID of the GPU to monitor.
        
    Returns:
        The GPU utilization percentage (0.0-100.0).
    """
    try:
        result = subprocess.run(
            ['nvidia-smi', 
             '--query-gpu=utilization.gpu', 
             '--format=csv,noheader,nounits',
             f'--id={gpu_id}'], 
            capture_output=True, text=True, check=True
        )
        return float(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError, FileNotFoundError) as e:
        logging.error(f"Failed to get GPU utilization for device {gpu_id}: {e}")
        return 0.0

def get_gpu_memory(gpu_id: int) -> tuple:
    """Get GPU memory usage and total memory."""
    try:
        result = subprocess.run(
            ['nvidia-smi', 
             '--query-gpu=memory.used,memory.total', 
             '--format=csv,noheader,nounits',
             f'--id={gpu_id}'], 
            capture_output=True, text=True, check=True
        )
        used, total = result.stdout.strip().split(', ')
        return float(used) / 1024, float(total) / 1024  # Convert MB to GB
    except (subprocess.CalledProcessError, ValueError, FileNotFoundError) as e:
        logging.error(f"Failed to get GPU memory for device {gpu_id}: {e}")
        return 0.0, 0.0

def get_system_stats() -> tuple:
    """Get CPU and system memory usage."""
    cpu_usage = psutil.cpu_percent(interval=None)
    memory = psutil.virtual_memory()
    memory_used = memory.used / (1024**3)  # GB
    memory_total = memory.total / (1024**3)  # GB
    return cpu_usage, memory_used, memory_total

class GPUMonitor:
    """
    Enhanced GPU and system monitoring with data collection for visualization.
    """
    def __init__(self, gpu_ids: List[int], interval: int = 2, save_path: str = None):
        self.gpu_ids = gpu_ids
        self.interval = interval
        self.save_path = save_path
        self._thread = None
        self._is_running = False
        self._logger = setup_logger(__name__)
        
        # ✅ Data storage for visualization
        self.monitoring_data = {
            'timestamps': [],
            'cpu_usage': [],
            'memory_usage': [],
            'memory_total': [],
            'gpu_count': len(gpu_ids)
        }
        
        # Add per-GPU data storage
        for gpu_id in gpu_ids:
            self.monitoring_data[f'gpu_{gpu_id}_usage'] = []
            self.monitoring_data[f'gpu_{gpu_id}_memory'] = []
            self.monitoring_data[f'gpu_{gpu_id}_memory_total'] = []
        
        self._start_time = None

    def _monitor_loop(self):
        """Enhanced monitoring loop that collects data for visualization."""
        self._start_time = time.time()
        
        while self._is_running:
            try:
                current_time = time.time() - self._start_time
                
                # Get system stats
                cpu_usage, memory_used, memory_total = get_system_stats()
                
                # Store system data
                self.monitoring_data['timestamps'].append(current_time)
                self.monitoring_data['cpu_usage'].append(cpu_usage)
                self.monitoring_data['memory_usage'].append(memory_used)
                self.monitoring_data['memory_total'].append(memory_total)
                
                # Get GPU stats
                gpu_stats = {}
                for gpu_id in self.gpu_ids:
                    utilization = get_gpu_utilization(gpu_id)
                    gpu_memory_used, gpu_memory_total = get_gpu_memory(gpu_id)
                    
                    # Store GPU data
                    self.monitoring_data[f'gpu_{gpu_id}_usage'].append(utilization)
                    self.monitoring_data[f'gpu_{gpu_id}_memory'].append(gpu_memory_used)
                    self.monitoring_data[f'gpu_{gpu_id}_memory_total'].append(gpu_memory_total)
                    
                    gpu_stats[f'gpu_{gpu_id}'] = {
                        'utilization': utilization,
                        'memory_used': gpu_memory_used,
                        'memory_total': gpu_memory_total
                    }
                
                # Log current stats
                self._logger.debug(f"System: CPU {cpu_usage:.1f}%, RAM {memory_used:.1f}GB")
                for gpu_id, stats in gpu_stats.items():
                    self._logger.debug(f"{gpu_id}: {stats['utilization']:.1f}% util, "
                                     f"{stats['memory_used']:.1f}GB mem")
                
                time.sleep(self.interval)
                
            except Exception as e:
                self._logger.error(f"Error in monitoring loop: {e}")
                break

    def start(self):
        """Starts the enhanced monitoring thread."""
        if self._thread is not None and self._thread.is_alive():
            self._logger.warning("Monitoring thread is already running.")
            return

        self._is_running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        self._logger.info(f"Started monitoring for GPUs: {self.gpu_ids}")

    def stop(self):
        """Stops monitoring and optionally saves data."""
        if self._thread is None or not self._thread.is_alive():
            return
        
        self._is_running = False
        self._thread.join()
        self._logger.info("Stopped GPU monitoring.")
        
        # Save data if path provided
        if self.save_path:
            self.save_monitoring_data()

    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics from collected data."""
        if not self.monitoring_data['timestamps']:
            return {}
        
        duration = max(self.monitoring_data['timestamps']) if self.monitoring_data['timestamps'] else 0
        
        summary = {
            'duration_minutes': duration / 60,
            'cpu_usage': {
                'mean': sum(self.monitoring_data['cpu_usage']) / len(self.monitoring_data['cpu_usage']),
                'max': max(self.monitoring_data['cpu_usage']),
                'min': min(self.monitoring_data['cpu_usage'])
            } if self.monitoring_data['cpu_usage'] else {},
            'memory_usage': {
                'mean_gb': sum(self.monitoring_data['memory_usage']) / len(self.monitoring_data['memory_usage']),
                'max_gb': max(self.monitoring_data['memory_usage'])
            } if self.monitoring_data['memory_usage'] else {}
        }
        
        # Add GPU summaries
        for gpu_id in self.gpu_ids:
            usage_key = f'gpu_{gpu_id}_usage'
            memory_key = f'gpu_{gpu_id}_memory'
            
            if usage_key in self.monitoring_data and self.monitoring_data[usage_key]:
                summary[f'gpu_{gpu_id}'] = {
                    'usage_mean': sum(self.monitoring_data[usage_key]) / len(self.monitoring_data[usage_key]),
                    'usage_max': max(self.monitoring_data[usage_key]),
                    'memory_mean_gb': sum(self.monitoring_data[memory_key]) / len(self.monitoring_data[memory_key]),
                    'memory_max_gb': max(self.monitoring_data[memory_key])
                }
        
        return summary

    def save_monitoring_data(self):
        """Save monitoring data to JSON file."""
        if not self.save_path:
            return
        
        save_data = {
            'hardware_usage': self.monitoring_data,
            'summary_stats': self.get_summary_stats(),
            'monitoring_interval': self.interval,
            'gpu_count': len(self.gpu_ids),
            'gpu_ids': self.gpu_ids
        }
        
        try:
            with open(self.save_path, 'w') as f:
                json.dump(save_data, f, indent=2)
            self._logger.info(f"💾 Hardware monitoring data saved to: {self.save_path}")
        except Exception as e:
            self._logger.error(f"Failed to save monitoring data: {e}")

# ✅ Context manager using existing GPUMonitor
class GPUMonitorContext:
    """Context manager for the existing GPU monitoring system."""
    
    def __init__(self, gpu_ids: List[int] = None, interval: int = 2, save_path: str = None):
        if gpu_ids is None:
            # Auto-detect GPUs
            gpu_ids = self._detect_gpus()
        
        self.monitor = GPUMonitor(gpu_ids, interval, save_path)
    
    def _detect_gpus(self) -> List[int]:
        """Auto-detect available GPUs."""
        try:
            result = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True)
            gpu_count = len([line for line in result.stdout.split('\n') if 'GPU' in line])
            return list(range(gpu_count))
        except:
            return [0]  # Assume at least one GPU
    
    def __enter__(self):
        self.monitor.start()
        return self.monitor
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.monitor.stop()