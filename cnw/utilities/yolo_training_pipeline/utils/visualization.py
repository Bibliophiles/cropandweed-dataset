#!/usr/bin/env python3
"""
Visualization utilities for YOLO training pipeline results.
Generates plots for metrics, hardware usage, and hyperparameter analysis.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

# Add missing imports for existing modules
from .gpu_monitoring import GPUMonitor, get_gpu_utilization
from .logging_utils import setup_logger

# Set matplotlib style
try:
    plt.style.use('seaborn-v0_8')
except OSError:
    # Fallback if seaborn style not available
    plt.style.use('default')
    
sns.set_palette("husl")

class PipelineVisualizer:
    """Creates visualizations for pipeline results and metrics."""
    
    def __init__(self, output_dir: str, experiment_name: str = "experiment"):
        self.output_dir = Path(output_dir).resolve()
        self.viz_dir = self.output_dir / "visualizations"
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name
        self.logger = setup_logger(__name__)
        
        # Configure matplotlib for better plots
        plt.rcParams.update({
            'figure.figsize': (12, 8),
            'font.size': 10,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 16
        })
    
    def load_results_data(self, results_file: str) -> Dict[str, Any]:
        """Load results data from JSON file."""
        try:
            with open(results_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load results from {results_file}: {e}")
            return {}
    
    def load_metrics_csv(self, csv_file: str) -> pd.DataFrame:
        """Load metrics from CSV file."""
        try:
            return pd.read_csv(csv_file)
        except Exception as e:
            self.logger.error(f"Failed to load CSV from {csv_file}: {e}")
            return pd.DataFrame()
    
    def plot_map_metrics(self, data: Dict[str, Any], save: bool = True) -> plt.Figure:
        """Plot mAP metrics and performance analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Performance Metrics - {self.experiment_name}', fontsize=16, fontweight='bold')
        
        # Extract nested CV results if available
        individual_scores = data.get('individual_scores', [])
        mean_score = data.get('mean_test_score', 0)
        std_score = data.get('std_test_score', 0)
        
        if individual_scores:
            # Plot 1: Individual fold scores
            folds = list(range(1, len(individual_scores) + 1))
            axes[0, 0].bar(folds, individual_scores, alpha=0.7, color='skyblue')
            axes[0, 0].axhline(y=mean_score, color='red', linestyle='--', 
                              label=f'Mean: {mean_score:.3f}')
            axes[0, 0].set_title('Cross-Validation Scores by Fold')
            axes[0, 0].set_xlabel('Fold')
            axes[0, 0].set_ylabel('mAP Score')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Plot 2: Score distribution
            axes[0, 1].hist(individual_scores, bins=max(3, len(individual_scores)//2), 
                           alpha=0.7, color='lightgreen', edgecolor='black')
            axes[0, 1].axvline(x=mean_score, color='red', linestyle='--', 
                              label=f'Mean: {mean_score:.3f}')
            axes[0, 1].set_title('Score Distribution')
            axes[0, 1].set_xlabel('mAP Score')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].legend()
            
            # Plot 3: Performance stability
            axes[1, 0].plot(folds, individual_scores, 'o-', linewidth=2, markersize=8)
            axes[1, 0].fill_between(folds, 
                                   [mean_score - std_score] * len(folds),
                                   [mean_score + std_score] * len(folds),
                                   alpha=0.2, color='red', label='±1 std')
            axes[1, 0].axhline(y=mean_score, color='red', linestyle='--')
            axes[1, 0].set_title('Performance Stability Across Folds')
            axes[1, 0].set_xlabel('Fold')
            axes[1, 0].set_ylabel('mAP Score')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # Plot 4: Summary statistics
            stats_text = f"""
Performance Summary:
• Mean mAP: {mean_score:.4f}
• Std Dev: ±{std_score:.4f}
• Best Fold: {max(individual_scores):.4f}
• Worst Fold: {min(individual_scores):.4f}
• CV Coefficient: {(std_score/mean_score*100):.1f}%
            """
            axes[1, 1].text(0.1, 0.5, stats_text, fontsize=12, 
                            verticalalignment='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Performance Statistics')
            axes[1, 1].axis('off')
        else:
            # No CV data available
            for ax in axes.flat:
                ax.text(0.5, 0.5, 'Cross-Validation\nResults Not Available', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=14)
                ax.set_title('Performance Metrics')
        
        plt.tight_layout()
        
        if save:
            try:
                save_path = self.viz_dir / f"{self.experiment_name}_performance_metrics.png"
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Performance metrics plot saved to: {save_path}")
            except Exception as e:
                self.logger.error(f"Failed to save performance plot: {e}")
        
        return fig
    
    def plot_hyperparameter_analysis(self, data: Dict[str, Any], save: bool = True) -> plt.Figure:
        """Plot hyperparameter analysis and optimization results."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Hyperparameter Analysis - {self.experiment_name}', fontsize=16, fontweight='bold')
        
        # Extract hyperparameter data if available
        best_params = data.get('best_hyperparameters', {})
        optuna_trials = data.get('optuna_trials', [])
        
        if best_params:
            # Plot 1: Best hyperparameters
            param_names = list(best_params.keys())
            param_values = list(best_params.values())
            
            axes[0, 0].barh(param_names, param_values, alpha=0.7, color='orange')
            axes[0, 0].set_title('Best Hyperparameters')
            axes[0, 0].set_xlabel('Parameter Value')
            
            # Plot 2: Parameter importance (placeholder)
            if len(param_names) > 1:
                importance = [1.0 / len(param_names)] * len(param_names)  # Placeholder
                axes[0, 1].pie(importance, labels=param_names, autopct='%1.1f%%', startangle=90)
                axes[0, 1].set_title('Parameter Importance (Estimated)')
            else:
                axes[0, 1].text(0.5, 0.5, 'Parameter Importance\nNot Available', 
                                ha='center', va='center', transform=axes[0, 1].transAxes)
                axes[0, 1].set_title('Parameter Importance')
        
        if optuna_trials:
            # Plot 3: Optimization progress
            trial_numbers = [trial.get('number', i) for i, trial in enumerate(optuna_trials)]
            trial_values = [trial.get('value', 0) for trial in optuna_trials]
            
            axes[1, 0].plot(trial_numbers, trial_values, 'o-', alpha=0.7)
            axes[1, 0].set_title('Optimization Progress')
            axes[1, 0].set_xlabel('Trial Number')
            axes[1, 0].set_ylabel('Objective Value')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Plot 4: Trial value distribution
            axes[1, 1].hist(trial_values, bins=min(20, len(trial_values)//2), 
                           alpha=0.7, color='lightblue', edgecolor='black')
            axes[1, 1].set_title('Trial Value Distribution')
            axes[1, 1].set_xlabel('Objective Value')
            axes[1, 1].set_ylabel('Frequency')
        else:
            # No optimization data
            for ax in [axes[1, 0], axes[1, 1]]:
                ax.text(0.5, 0.5, 'Hyperparameter\nOptimization Data\nNot Available', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
        
        plt.tight_layout()
        
        if save:
            try:
                save_path = self.viz_dir / f"{self.experiment_name}_hyperparameters.png"
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Hyperparameter analysis plot saved to: {save_path}")
            except Exception as e:
                self.logger.error(f"Failed to save hyperparameter plot: {e}")
        
        return fig
    
    def plot_hardware_usage(self, data: Dict[str, Any], save: bool = True) -> plt.Figure:
        """Plot hardware usage during training."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Hardware Usage Analysis - {self.experiment_name}', fontsize=16, fontweight='bold')
        
        # Extract hardware usage data if available
        if 'hardware_usage' in data:
            usage_data = data['hardware_usage']
            
            # CPU Usage
            if 'cpu_usage' in usage_data and usage_data['cpu_usage']:
                timestamps = usage_data.get('timestamps', list(range(len(usage_data['cpu_usage']))))
                cpu_data = usage_data['cpu_usage']
                
                axes[0, 0].plot(timestamps, cpu_data, linewidth=2, color='blue', label='CPU Usage')
                axes[0, 0].set_title('CPU Utilization Over Time')
                axes[0, 0].set_xlabel('Time (seconds)')
                axes[0, 0].set_ylabel('CPU Usage (%)')
                axes[0, 0].grid(True, alpha=0.3)
                axes[0, 0].fill_between(timestamps, cpu_data, alpha=0.3, color='blue')
                
                mean_cpu = np.mean(cpu_data)
                axes[0, 0].axhline(y=mean_cpu, color='red', linestyle='--', alpha=0.7, 
                                  label=f'Mean: {mean_cpu:.1f}%')
                axes[0, 0].legend()
            else:
                axes[0, 0].text(0.5, 0.5, 'CPU Usage Data\nNot Available', 
                               ha='center', va='center', transform=axes[0, 0].transAxes, fontsize=12)
                axes[0, 0].set_title('CPU Usage')
            
            # Memory Usage  
            if 'memory_usage' in usage_data and usage_data['memory_usage']:
                timestamps = usage_data.get('timestamps', list(range(len(usage_data['memory_usage']))))
                memory_data = usage_data['memory_usage']
                memory_total = usage_data.get('memory_total', [])
                
                axes[0, 1].plot(timestamps, memory_data, linewidth=2, color='orange', label='Memory Used')
                
                if memory_total:
                    axes[0, 1].axhline(y=memory_total[0], color='gray', linestyle=':', alpha=0.7, 
                                      label=f'Total: {memory_total[0]:.1f} GB')
                
                axes[0, 1].set_title('System Memory Usage Over Time')
                axes[0, 1].set_xlabel('Time (seconds)')
                axes[0, 1].set_ylabel('Memory Usage (GB)')
                axes[0, 1].grid(True, alpha=0.3)
                axes[0, 1].fill_between(timestamps, memory_data, alpha=0.3, color='orange')
                
                mean_memory = np.mean(memory_data)
                axes[0, 1].axhline(y=mean_memory, color='red', linestyle='--', alpha=0.7, 
                                  label=f'Mean: {mean_memory:.1f} GB')
                axes[0, 1].legend()
            else:
                axes[0, 1].text(0.5, 0.5, 'Memory Usage Data\nNot Available', 
                               ha='center', va='center', transform=axes[0, 1].transAxes, fontsize=12)
                axes[0, 1].set_title('Memory Usage')
            
            # GPU Usage (first GPU)
            gpu_usage_key = 'gpu_0_usage'
            if gpu_usage_key in usage_data and usage_data[gpu_usage_key]:
                timestamps = usage_data.get('timestamps', list(range(len(usage_data[gpu_usage_key]))))
                gpu_data = usage_data[gpu_usage_key]
                
                axes[1, 0].plot(timestamps, gpu_data, linewidth=2, color='green', label='GPU Utilization')
                axes[1, 0].set_title('GPU Utilization Over Time')
                axes[1, 0].set_xlabel('Time (seconds)')
                axes[1, 0].set_ylabel('GPU Usage (%)')
                axes[1, 0].grid(True, alpha=0.3)
                axes[1, 0].fill_between(timestamps, gpu_data, alpha=0.3, color='green')
                
                mean_gpu = np.mean(gpu_data)
                axes[1, 0].axhline(y=mean_gpu, color='red', linestyle='--', alpha=0.7, 
                                  label=f'Mean: {mean_gpu:.1f}%')
                axes[1, 0].legend()
            else:
                axes[1, 0].text(0.5, 0.5, 'GPU Usage Data\nNot Available', 
                               ha='center', va='center', transform=axes[1, 0].transAxes, fontsize=12)
                axes[1, 0].set_title('GPU Usage')
            
            # GPU Memory (first GPU)
            gpu_memory_key = 'gpu_0_memory'
            if gpu_memory_key in usage_data and usage_data[gpu_memory_key]:
                timestamps = usage_data.get('timestamps', list(range(len(usage_data[gpu_memory_key]))))
                gpu_mem_data = usage_data[gpu_memory_key]
                gpu_mem_total = usage_data.get('gpu_0_memory_total', [])
                
                axes[1, 1].plot(timestamps, gpu_mem_data, linewidth=2, color='red', label='GPU Memory Used')
                
                if gpu_mem_total:
                    axes[1, 1].axhline(y=gpu_mem_total[0], color='gray', linestyle=':', alpha=0.7, 
                                      label=f'Total: {gpu_mem_total[0]:.1f} GB')
                
                axes[1, 1].set_title('GPU Memory Usage Over Time')
                axes[1, 1].set_xlabel('Time (seconds)')
                axes[1, 1].set_ylabel('GPU Memory (GB)')
                axes[1, 1].grid(True, alpha=0.3)
                axes[1, 1].fill_between(timestamps, gpu_mem_data, alpha=0.3, color='red')
                
                mean_gpu_mem = np.mean(gpu_mem_data)
                axes[1, 1].axhline(y=mean_gpu_mem, color='darkred', linestyle='--', alpha=0.7, 
                                  label=f'Mean: {mean_gpu_mem:.1f} GB')
                axes[1, 1].legend()
            else:
                axes[1, 1].text(0.5, 0.5, 'GPU Memory Data\nNot Available', 
                               ha='center', va='center', transform=axes[1, 1].transAxes, fontsize=12)
                axes[1, 1].set_title('GPU Memory')
        else:
            # No hardware data available
            for i, ax in enumerate(axes.flat):
                ax.text(0.5, 0.5, 'Hardware Usage Data\nNot Available\n(Hardware monitoring ready)', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
                ax.set_title(['CPU Usage', 'Memory Usage', 'GPU Usage', 'GPU Memory'][i])
        
        plt.tight_layout()
        
        if save:
            try:
                save_path = self.viz_dir / f"{self.experiment_name}_hardware_usage.png"
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Hardware usage plot saved to: {save_path}")
            except Exception as e:
                self.logger.error(f"Failed to save hardware plot: {e}")
        
        return fig
    
    def create_summary_report(self, data: Dict[str, Any], save: bool = True) -> plt.Figure:
        """Create a comprehensive summary report."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Training Pipeline Summary - {self.experiment_name}', fontsize=18, fontweight='bold')
        
        # Extract key metrics
        mean_score = data.get('mean_test_score', 0)
        std_score = data.get('std_test_score', 0)
        individual_scores = data.get('individual_scores', [])
        best_params = data.get('best_hyperparameters', {})
        hardware_summary = data.get('hardware_summary', {})
        
        # Plot 1: Performance overview
        if individual_scores:
            folds = list(range(1, len(individual_scores) + 1))
            axes[0, 0].bar(folds, individual_scores, alpha=0.7, color='lightblue')
            axes[0, 0].axhline(y=mean_score, color='red', linestyle='--', linewidth=2)
            axes[0, 0].set_title('Cross-Validation Performance')
            axes[0, 0].set_xlabel('Fold')
            axes[0, 0].set_ylabel('mAP Score')
            axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Key metrics summary
        summary_text = f"""
TRAINING RESULTS SUMMARY

Performance Metrics:
• Mean mAP: {mean_score:.4f}
• Standard Deviation: ±{std_score:.4f}
• Coefficient of Variation: {(std_score/mean_score*100) if mean_score > 0 else 0:.1f}%

Cross-Validation:
• Number of Folds: {len(individual_scores)}
• Best Fold Score: {max(individual_scores) if individual_scores else 'N/A'}
• Worst Fold Score: {min(individual_scores) if individual_scores else 'N/A'}

Best Hyperparameters:
"""
        
        for param, value in list(best_params.items())[:5]:  # Show top 5 params
            summary_text += f"• {param}: {value}\n"
        
        if hardware_summary:
            duration = hardware_summary.get('duration_minutes', 0)
            summary_text += f"\nTraining Duration: {duration:.1f} minutes"
        
        axes[0, 1].text(0.05, 0.95, summary_text, fontsize=10, verticalalignment='top',
                        transform=axes[0, 1].transAxes, fontfamily='monospace')
        axes[0, 1].set_title('Experiment Summary')
        axes[0, 1].axis('off')
        
        # Plot 3: Hardware utilization overview
        if hardware_summary and 'gpu_0' in hardware_summary:
            gpu_stats = hardware_summary['gpu_0']
            metrics = ['GPU Usage', 'GPU Memory', 'CPU Usage', 'System Memory']
            values = [
                gpu_stats.get('usage_mean', 0),
                gpu_stats.get('memory_mean_gb', 0) * 10,  # Scale for visualization
                hardware_summary.get('cpu_usage', {}).get('mean', 0),
                hardware_summary.get('memory_usage', {}).get('mean_gb', 0)
            ]
            
            colors = ['green', 'red', 'blue', 'orange']
            axes[1, 0].bar(metrics, values, alpha=0.7, color=colors)
            axes[1, 0].set_title('Average Hardware Utilization')
            axes[1, 0].set_ylabel('Usage (%)')
            axes[1, 0].tick_params(axis='x', rotation=45)
        else:
            axes[1, 0].text(0.5, 0.5, 'Hardware Data\nNot Available', 
                           ha='center', va='center', transform=axes[1, 0].transAxes, fontsize=14)
            axes[1, 0].set_title('Hardware Utilization')
        
        # Plot 4: Configuration details
        config_text = f"""
EXPERIMENT CONFIGURATION

Model: {data.get('model_name', 'N/A')}
Dataset: {data.get('dataset_name', 'N/A')}

Optimization Settings:
• Algorithm: Optuna (TPE)
• Nested CV: {'Yes' if individual_scores else 'No'}
• Folds: {len(individual_scores) if individual_scores else 'N/A'}

System Information:
• Python: {data.get('python_version', 'N/A')}
• PyTorch: {data.get('pytorch_version', 'N/A')}
• CUDA Available: {data.get('cuda_available', 'N/A')}

Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        axes[1, 1].text(0.05, 0.95, config_text, fontsize=9, verticalalignment='top',
                        transform=axes[1, 1].transAxes, fontfamily='monospace')
        axes[1, 1].set_title('Configuration Details')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save:
            try:
                save_path = self.viz_dir / f"{self.experiment_name}_summary_report.png"
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Summary report saved to: {save_path}")
            except Exception as e:
                self.logger.error(f"Failed to save summary report: {e}")
        
        return fig
    
    def generate_all_plots(self, results_file: str, experiment_name: Optional[str] = None) -> None:
        """Generate all visualization plots from results file."""
        if experiment_name:
            self.experiment_name = experiment_name
        
        # Load data
        data = self.load_results_data(results_file)
        
        if not data:
            self.logger.warning("No data found for visualization")
            return
        
        try:
            # Generate all plots
            self.logger.info("📊 Generating performance metrics plot...")
            self.plot_map_metrics(data)
            
            self.logger.info("📊 Generating hyperparameter analysis plot...")
            self.plot_hyperparameter_analysis(data)
            
            self.logger.info("📊 Generating hardware usage plot...")
            self.plot_hardware_usage(data)
            
            self.logger.info("📊 Generating summary report...")
            self.create_summary_report(data)
            
            self.logger.info(f"✅ All plots generated successfully in: {self.viz_dir}")
            
        except Exception as e:
            self.logger.error(f"Failed to generate plots: {e}")
            import traceback
            self.logger.debug(f"Full traceback: {traceback.format_exc()}")

def create_visualization_from_results(results_file: str, output_dir: str, experiment_name: str = "experiment"):
    """
    Convenience function to create all visualizations from results file.
    """
    visualizer = PipelineVisualizer(output_dir, experiment_name)
    visualizer.generate_all_plots(results_file, experiment_name)
    return visualizer.viz_dir