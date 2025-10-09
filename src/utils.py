import numpy as np
import pandas as pd
import json
import pickle
import joblib
import os
import time
import logging
from datetime import datetime
from pathlib import Path

def setup_logging(log_level=logging.INFO, log_file=None):
    """
    Set up logging configuration.
    
    Args:
        log_level: Logging level
        log_file (str, optional): Path to log file
    """
    format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    if log_file:
        logging.basicConfig(
            level=log_level,
            format=format_string,
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    else:
        logging.basicConfig(level=log_level, format=format_string)


def create_directory_structure(base_dir='./'):
    """
    Create the project directory structure. This creates the folders in the root directory
    
    Args:
        base_dir (str): Base directory path
    """
    directories = [
        'data',
        'results',
        'results/figures',
        'results/metrics',
        'results/models',
        'logs'
    ]
    
    for directory in directories:
        Path(os.path.join(base_dir, directory)).mkdir(parents=True, exist_ok=True)
    
    logging.info(f"Created directory structure in {base_dir}")

def save_results(results, filepath, format='pickle'):
    """
    Save results to file.
    
    Args:
        results: Results object to save
        filepath (str): Path to save file
        format (str): Save format ('pickle', 'joblib', 'json')
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    if format == 'pickle':
        with open(filepath, 'wb') as f:
            pickle.dump(results, f)
    elif format == 'joblib':
        joblib.dump(results, filepath)
    elif format == 'json':
        # Convert numpy arrays to lists for JSON serialization
        json_results = convert_for_json(results)
        with open(filepath, 'w') as f:
            json.dump(json_results, f, indent=2)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    logging.info(f"Saved results to {filepath}")

def load_results(filepath, format='auto'):
    """
    Load results from file.
    
    Args:
        filepath (str): Path to load file
        format (str): File format ('pickle', 'joblib', 'json', 'auto')
        
    Returns:
        Loaded results object
    """
    if format == 'auto':
        format = Path(filepath).suffix[1:]  # Get extension without dot
        
    if format in ['pkl', 'pickle']:
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    elif format in ['joblib', 'jlib']:
        return joblib.load(filepath)
    elif format == 'json':
        with open(filepath, 'r') as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
def convert_for_json(obj):
    """
    Convert numpy arrays and other non-JSON-serializable objects.
    
    Args:
        obj: Object to convert
        
    Returns:
        JSON-serializable object
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {key: convert_for_json(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_for_json(item) for item in obj]
    else:
        return obj
    
class Timer:
    """Context manager for timing operations."""
    
    def __init__(self, description="Operation"):
        self.description = description
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        logging.info(f"Starting {self.description}...")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start_time
        logging.info(f"Completed {self.description} in {elapsed:.2f} seconds")


class ExperimentLogger:
    """
    Logger for tracking experiments and results.
    """
    
    def __init__(self, log_dir='logs/'):
        self.log_dir = log_dir
        self.experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f'experiment_{self.experiment_id}.json')
        self.experiments = []
        os.makedirs(log_dir, exist_ok=True)
    
    def log_experiment(self, method_name, parameters, results, metrics=None):
        """
        Log an experiment.
        
        Args:
            method_name (str): Name of the method
            parameters (dict): Method parameters
            results (dict): Results from the method
            metrics (dict, optional): Evaluation metrics
        """
        experiment = {
            'timestamp': datetime.now().isoformat(),
            'method': method_name,
            'parameters': parameters,
            'results': convert_for_json(results),
            'metrics': convert_for_json(metrics) if metrics else None
        }
        
        self.experiments.append(experiment)
        
        # Save to file
        with open(self.log_file, 'w') as f:
            json.dump(self.experiments, f, indent=2)
    
    def get_best_results(self, metric='silhouette_score', ascending=False):
        """
        Get best results based on a specific metric.
        
        Args:
            metric (str): Metric to optimize
            ascending (bool): Sort order
            
        Returns:
            dict: Best experiment results
        """
        scored_experiments = []
        
        for exp in self.experiments:
            if exp.get('metrics') and metric in exp['metrics']:
                scored_experiments.append({
                    'experiment': exp,
                    'score': exp['metrics'][metric]
                })
        
        if not scored_experiments:
            return None
        
        sorted_experiments = sorted(scored_experiments, 
                                  key=lambda x: x['score'], 
                                  reverse=not ascending)
        
        return sorted_experiments[0]['experiment']

def validate_input_data(X, y=None, min_samples=10, max_features=10000):
    """
    Validate input data for dimensionality reduction.
    
    Args:
        X (np.ndarray): Input features
        y (np.ndarray, optional): Labels
        min_samples (int): Minimum number of samples
        max_features (int): Maximum number of features
        
    Returns:
        tuple: (is_valid, error_message)
    """
    if not isinstance(X, np.ndarray):
        return False, "Input X must be a numpy array"
    
    if X.ndim != 2:
        return False, "Input X must be 2-dimensional"
    
    if X.shape[0] < min_samples:
        return False, f"Need at least {min_samples} samples, got {X.shape[0]}"
    
    if X.shape[1] > max_features:
        return False, f"Too many features: {X.shape[1]} > {max_features}"
    
    if y is not None:
        if not isinstance(y, np.ndarray):
            return False, "Labels y must be a numpy array"
        
        if len(y) != X.shape[0]:
            return False, "Number of labels must match number of samples"
    
    if np.any(np.isnan(X)):
        return False, "Input data contains NaN values"
    
    if np.any(np.isinf(X)):
        return False, "Input data contains infinite values"
    
    return True, "Data is valid"

def get_memory_usage():
    """
    Get current memory usage.
    
    Returns:
        dict: Memory usage information
    """
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    return {
        'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size
        'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size
        'percent': process.memory_percent()
    }

def estimate_memory_requirement(n_samples, n_features, method='pca'):
    """
    Estimate memory requirement for dimensionality reduction.
    
    Args:
        n_samples (int): Number of samples
        n_features (int): Number of features
        method (str): Dimensionality reduction method
        
    Returns:
        float: Estimated memory in MB
    """
    # Base data size (float64)
    base_memory = n_samples * n_features * 8 / 1024 / 1024
    
    # Method-specific multipliers
    multipliers = {
        'pca': 3.0,      # Data + covariance matrix + components
        'tsne': 2.5,     # Data + distance matrices
        'umap': 2.0,     # Data + graph structures
        'lda': 2.5,      # Data + class statistics
        'ica': 3.0,      # Data + unmixing matrices
        'kernel_pca': 4.0,  # Data + kernel matrix
        'isomap': 3.5,   # Data + distance matrices
        'lle': 3.0,      # Data + neighbor matrices
        'autoencoder': 5.0  # Data + network parameters
    }
    
    multiplier = multipliers.get(method, 3.0)
    estimated_memory = base_memory * multiplier
    
    return estimated_memory

def check_system_requirements(n_samples, n_features, methods):
    """
    Check if system can handle the computation.
    
    Args:
        n_samples (int): Number of samples
        n_features (int): Number of features  
        methods (list): List of methods to run
        
    Returns:
        dict: System requirement check results
    """
    import psutil
    
    # Get system info
    total_memory = psutil.virtual_memory().total / 1024 / 1024  # MB
    available_memory = psutil.virtual_memory().available / 1024 / 1024  # MB
    
    # Estimate requirements
    max_memory_needed = 0
    method_estimates = {}
    
    for method in methods:
        estimate = estimate_memory_requirement(n_samples, n_features, method)
        method_estimates[method] = estimate
        max_memory_needed = max(max_memory_needed, estimate)
    
    # Check feasibility
    feasible = max_memory_needed < available_memory * 0.8  # Leave 20% buffer
    
    return {
        'total_memory_mb': total_memory,
        'available_memory_mb': available_memory,
        'max_memory_needed_mb': max_memory_needed,
        'method_estimates': method_estimates,
        'feasible': feasible,
        'recommendation': _get_memory_recommendation(feasible, max_memory_needed, available_memory)
    }

def _get_memory_recommendation(feasible, needed, available):
    """Generate memory usage recommendation."""
    if feasible:
        return "System resources are sufficient for all methods."
    else:
        ratio = needed / available
        if ratio < 1.5:
            return "Close to memory limit. Consider reducing sample size or running methods individually."
        else:
            return f"Insufficient memory. Need {needed:.0f}MB but only {available:.0f}MB available. Reduce data size significantly."


class ProgressTracker:
    """Track progress of long-running operations."""
    
    def __init__(self, total_steps, description="Processing"):
        self.total_steps = total_steps
        self.current_step = 0
        self.description = description
        self.start_time = time.time()
        
    def update(self, step_description=""):
        """Update progress."""
        self.current_step += 1
        elapsed = time.time() - self.start_time
        
        if self.current_step > 0:
            eta = elapsed * (self.total_steps - self.current_step) / self.current_step
            progress = self.current_step / self.total_steps * 100
            
            print(f"\r{self.description}: {progress:.1f}% ({self.current_step}/{self.total_steps}) "
                  f"- ETA: {eta:.1f}s - {step_description}", end="", flush=True)
        
        if self.current_step >= self.total_steps:
            print(f"\n{self.description} completed in {elapsed:.2f}s")


def create_hyperparameter_combinations(param_grid):
    """
    Create all combinations of hyperparameters.
    
    Args:
        param_grid (dict): Dictionary of parameter lists
        
    Returns:
        list: List of parameter combinations
    """
    from itertools import product
    
    if not param_grid:
        return [{}]
    
    keys = param_grid.keys()
    values = param_grid.values()
    
    combinations = []
    for combination in product(*values):
        combinations.append(dict(zip(keys, combination)))
    
    return combinations


def sample_data_for_visualization(X, y=None, max_samples=2000, random_state=42):
    """
    Sample data for visualization to avoid overcrowding.
    
    Args:
        X (np.ndarray): Input data
        y (np.ndarray, optional): Labels
        max_samples (int): Maximum number of samples to keep
        random_state (int): Random seed
        
    Returns:
        tuple: (X_sampled, y_sampled) or X_sampled if y is None
    """
    if X.shape[0] <= max_samples:
        return (X, y) if y is not None else X
    
    np.random.seed(random_state)
    indices = np.random.choice(X.shape[0], max_samples, replace=False)
    
    X_sampled = X[indices]
    
    if y is not None:
        y_sampled = y[indices]
        return X_sampled, y_sampled
    
    return X_sampled


def generate_summary_report(results_dict, evaluations, output_path=None):
    """
    Generate a comprehensive summary report.
    
    Args:
        results_dict (dict): Dimensionality reduction results
        evaluations (dict): Evaluation results
        output_path (str, optional): Path to save report
        
    Returns:
        str: Summary report text
    """
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("DIMENSIONALITY REDUCTION COMPARISON REPORT")
    report_lines.append("=" * 80)
    report_lines.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    # Dataset summary
    if results_dict:
        first_result = next(iter(results_dict.values()))
        if 'embedding' in first_result:
            original_dim = "Unknown"
            reduced_dim = first_result['embedding'].shape[1]
            n_samples = first_result['embedding'].shape[0]
            
            report_lines.append("DATASET SUMMARY")
            report_lines.append("-" * 40)
            report_lines.append(f"Number of samples: {n_samples}")
            report_lines.append(f"Original dimensions: {original_dim}")
            report_lines.append(f"Reduced dimensions: {reduced_dim}")
            report_lines.append("")
    
    # Method comparison
    report_lines.append("METHOD COMPARISON")
    report_lines.append("-" * 40)
    
    # Performance summary
    performance_data = []
    for method, eval_dict in evaluations.items():
        if 'error' not in eval_dict:
            performance_data.append({
                'Method': method,
                'Fit Time': eval_dict.get('fit_time', 'N/A'),
                'Silhouette': eval_dict.get('silhouette_score', 'N/A'),
                'Trustworthiness': eval_dict.get('trustworthiness', 'N/A')
            })
    
    if performance_data:
        df = pd.DataFrame(performance_data)
        report_lines.append(df.to_string(index=False))
        report_lines.append("")
    
    # Recommendations
    report_lines.append("RECOMMENDATIONS")
    report_lines.append("-" * 40)
    
    # Find best methods for different criteria
    best_speed = min(evaluations.items(), 
                    key=lambda x: x[1].get('fit_time', float('inf')) if 'error' not in x[1] else float('inf'))
    
    best_clusters = max(evaluations.items(), 
                       key=lambda x: x[1].get('silhouette_score', -1) if 'error' not in x[1] else -1)
    
    best_preservation = max(evaluations.items(), 
                           key=lambda x: x[1].get('trustworthiness', -1) if 'error' not in x[1] else -1)
    
    if 'error' not in best_speed[1]:
        report_lines.append(f"Fastest method: {best_speed[0]} ({best_speed[1].get('fit_time', 'N/A'):.2f}s)")
    
    if 'error' not in best_clusters[1]:
        report_lines.append(f"Best clustering: {best_clusters[0]} (Silhouette: {best_clusters[1].get('silhouette_score', 'N/A'):.3f})")
    
    if 'error' not in best_preservation[1]:
        report_lines.append(f"Best preservation: {best_preservation[0]} (Trustworthiness: {best_preservation[1].get('trustworthiness', 'N/A'):.3f})")
    
    report_lines.append("")
    report_lines.append("Use cases:")
    report_lines.append("- For visualization: t-SNE or UMAP")
    report_lines.append("- For speed: PCA or SVD")
    report_lines.append("- For downstream ML: LDA (supervised) or PCA")
    report_lines.append("- For non-linear data: UMAP, t-SNE, or Kernel PCA")
    
    report_text = "\n".join(report_lines)
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(report_text)
        logging.info(f"Summary report saved to {output_path}")
    
    return report_text

def compare_embeddings_stability(reducer, X, y=None, n_runs=5, methods=None):
    """
    Compare stability of embeddings across multiple runs.
    
    Args:
        reducer: DimensionalityReducer instance
        X (np.ndarray): Input data
        y (np.ndarray, optional): Labels
        n_runs (int): Number of runs for stability test
        methods (list, optional): Methods to test
        
    Returns:
        dict: Stability results
    """
    if methods is None:
        methods = ['pca', 'tsne', 'umap']
    
    stability_results = {}
    
    for method in methods:
        embeddings = []
        
        for run in range(n_runs):
            # Add some randomness by using different random states
            result = reducer.fit_transform_method(
                X, y, method, n_components=2, 
                random_state=42 + run
            )
            
            if 'embedding' in result:
                embeddings.append(result['embedding'])
        
        if embeddings:
            # Calculate pairwise correlations between runs
            correlations = []
            for i in range(len(embeddings)):
                for j in range(i + 1, len(embeddings)):
                    # Flatten embeddings and calculate correlation
                    corr = np.corrcoef(embeddings[i].flatten(), 
                                     embeddings[j].flatten())[0, 1]
                    correlations.append(corr)
            
            stability_results[method] = {
                'mean_correlation': np.mean(correlations),
                'std_correlation': np.std(correlations),
                'n_runs': n_runs
            }
    
    return stability_results

if __name__ == "__main__":
    # Example usage
    
    # Setup logging
    setup_logging(log_file='logs/utils_test.log')
    
    # Create directory structure
    create_directory_structure()
    
    # Test timer
    with Timer("Test operation"):
        time.sleep(1)
    
    # Test memory estimation
    memory_est = estimate_memory_requirement(10000, 784, 'pca')
    print(f"Estimated memory for PCA: {memory_est:.2f} MB")
    
    # Test system requirements
    system_check = check_system_requirements(5000, 784, ['pca', 'tsne', 'umap'])
    print(f"System feasible: {system_check['feasible']}")
    print(f"Recommendation: {system_check['recommendation']}")
    
    # Test hyperparameter combinations
    param_grid = {
        'n_components': [2, 3],
        'method': ['pca', 'ica']
    }
    combinations = create_hyperparameter_combinations(param_grid)
    print(f"Generated {len(combinations)} parameter combinations")
    
    print("Utils module test completed!")
