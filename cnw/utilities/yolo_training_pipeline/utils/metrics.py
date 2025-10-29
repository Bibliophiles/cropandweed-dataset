#!/usr/bin/env python3

import numpy as np
import json
import os
import datetime
from pathlib import Path
from typing import Dict, List, Any

# ======================================================================
# Metrics Utility Functions
# ======================================================================

def calculate_confidence_interval(values: List[float], confidence: float = 0.95) -> Dict[str, float]:
    """
    Calculate confidence interval for a list of values.
    
    Args:
        values: List of numeric values.
        confidence: Confidence level (default: 0.95).
        
    Returns:
        Dictionary with mean, lower_bound, upper_bound, and margin_error.
    """
    if not values or len(values) < 2:
        return {'mean': np.mean(values) if values else 0.0, 
                'lower_bound': np.nan, 'upper_bound': np.nan, 'margin_error': np.nan}
    
    import scipy.stats as stats
    
    mean = np.mean(values)
    sem = stats.sem(values)  # Standard error of the mean
    n = len(values)
    
    # Calculate t-statistic for given confidence level
    alpha = 1 - confidence
    t_stat = stats.t.ppf(1 - alpha / 2, n - 1)
    
    margin_error = t_stat * sem
    
    return {
        'mean': float(mean),
        'lower_bound': float(mean - margin_error),
        'upper_bound': float(mean + margin_error),
        'margin_error': float(margin_error)
    }

def format_nested_cv_report(results: Dict[str, Any]) -> str:
    """
    Formats the complete nested CV results into a human-readable string report.
    
    Args:
        results: The dictionary returned by OptunaNestedCV.run_nested_cv().
        
    Returns:
        A formatted string summary.
    """
    report = [
        "===========================================================",
        "        YOLOv{-} Nested Cross-Validation Report              ",
        "===========================================================",
        f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "-----------------------------------------------------------",
        "Overall Performance (Unbiased Estimate)",
        "-----------------------------------------------------------",
        f"Mean Score (mAP50-95):  {results['mean_test_score']:.4f}",
        f"Std Deviation:          {results['std_test_score']:.4f}",
        "-----------------------------------------------------------",
        "Individual Outer Fold Results",
        "-----------------------------------------------------------"
    ]
    
    for fold_result in results['fold_results']:
        fold_idx = fold_result['fold'] + 1
        test_score = fold_result['test_score']
        best_params = fold_result['best_params']
        
        report.append(f"Fold {fold_idx}:")
        report.append(f"  Test Score (mAP50-95): {test_score:.4f}")
        report.append(f"  Best Hyperparameters:")
        for key, value in best_params.items():
            report.append(f"    - {key}: {value}")
        report.append("")
    
    report.append("===========================================================")
    
    return "\n".join(report)

def aggregate_metrics(metrics_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate metrics across multiple folds/runs.
    
    Args:
        metrics_list: List of metric dictionaries from different folds
        
    Returns:
        Dictionary with aggregated metrics (mean, std, etc.)
    """
    if not metrics_list:
        return {}
    
    # Extract numeric metrics
    aggregated = {}
    
    # Get all numeric keys from the first metrics dict
    numeric_keys = []
    for key, value in metrics_list[0].items():
        if isinstance(value, (int, float, np.number)):
            numeric_keys.append(key)
    
    # Aggregate each numeric metric
    for key in numeric_keys:
        values = [m[key] for m in metrics_list if key in m and isinstance(m[key], (int, float, np.number))]
        if values:
            aggregated[f"{key}_mean"] = np.mean(values)
            aggregated[f"{key}_std"] = np.std(values)
            aggregated[f"{key}_min"] = np.min(values)
            aggregated[f"{key}_max"] = np.max(values)
            
            # Add confidence interval
            ci = calculate_confidence_interval(values)
            aggregated[f"{key}_ci"] = ci
    
    # Add summary statistics
    aggregated['n_folds'] = len(metrics_list)
    aggregated['fold_indices'] = [m.get('fold', i) for i, m in enumerate(metrics_list)]
    
    return aggregated

def format_metrics_table(aggregated_metrics: Dict[str, Any]) -> str:
    """
    Format aggregated metrics into a readable table string.
    
    Args:
        aggregated_metrics: Dictionary from aggregate_metrics()
        
    Returns:
        Formatted string table
    """
    if not aggregated_metrics:
        return "No metrics to display"
    
    lines = []
    lines.append("┌─────────────────────────────────────────────────────────┐")
    lines.append("│                    AGGREGATED METRICS                   │")
    lines.append("├─────────────────────────────────────────────────────────┤")
    
    # Find all base metric names (without _mean, _std suffixes)
    base_metrics = set()
    for key in aggregated_metrics:
        if key.endswith('_mean'):
            base_metrics.add(key[:-5])
    
    for metric in sorted(base_metrics):
        mean_key = f"{metric}_mean"
        std_key = f"{metric}_std"
        ci_key = f"{metric}_ci"
        
        if mean_key in aggregated_metrics:
            mean_val = aggregated_metrics[mean_key]
            std_val = aggregated_metrics.get(std_key, 0)
            ci = aggregated_metrics.get(ci_key, {})
            
            lines.append(f"│ {metric.upper():<15} │ {mean_val:.4f} ± {std_val:.4f}         │")
            
            if ci and 'lower_bound' in ci:
                lines.append(f"│ {'':<15} │ 95% CI: [{ci['lower_bound']:.4f}, {ci['upper_bound']:.4f}] │")
    
    lines.append("└─────────────────────────────────────────────────────────┘")
    
    return "\n".join(lines)

# ======================================================================
# Metrics Tracker Class
# ======================================================================

class MetricsTracker:
    """A tracker for saving and loading nested cross-validation results."""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def save_experiment_results(self, model_name: str, results: Dict[str, Any]):
        """
        Save the complete results of a nested cross-validation run to a JSON file.
        
        Args:
            model_name: The name of the model being evaluated.
            results: The dictionary returned by OptunaNestedCV.run_nested_cv().
        """
        results_file = self.output_dir / f"{model_name}_nested_cv_results.json"
        
        # Add a timestamp to the results for tracking
        results['timestamp'] = datetime.datetime.now().isoformat()
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"✅ Nested CV results saved to: {results_file}")
    
    def save_nested_cv_results(self, model_name: str, results: Dict[str, Any]):
        """Alias for save_experiment_results for backward compatibility"""
        self.save_experiment_results(model_name, results)
    
    def load_nested_cv_results(self, model_name: str) -> Dict[str, Any]:
        """
        Load the results of a nested cross-validation run from a JSON file.
        
        Args:
            model_name: The name of the model to load results for.
            
        Returns:
            The loaded results dictionary.
        """
        results_file = self.output_dir / f"{model_name}_nested_cv_results.json"
        
        if not results_file.exists():
            raise FileNotFoundError(f"Results file not found: {results_file}")
            
        with open(results_file, 'r') as f:
            return json.load(f)

    def create_nested_cv_summary(self, model_name: str, filename: str = None) -> str:
        """
        Loads a nested CV results file and creates a formatted summary report.
        
        Args:
            model_name: The name of the model to generate a report for.
            filename: The name of the file to save the report to. If None, prints to console.
            
        Returns:
            The formatted string report.
        """
        try:
            results = self.load_nested_cv_results(model_name)
            report = format_nested_cv_report(results)
            
            if filename:
                report_file = self.output_dir / filename
                with open(report_file, 'w') as f:
                    f.write(report)
                print(f"📄 Summary report saved to: {report_file}")
            
            return report
            
        except FileNotFoundError as e:
            print(f"❌ Error: {e}")
            return ""