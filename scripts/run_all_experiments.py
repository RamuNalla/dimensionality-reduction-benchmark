import os
import sys
import argparse
import logging
from pathlib import Path
import json
import time

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_loader import load_fashion_mnist
from dimensionality_reducer import DimensionalityReducer
from evaluators import evaluate_all_metrics
from visualizers import save_all_plots
from utils import (
    setup_logging, create_directory_structure, save_results, 
    Timer, ExperimentLogger, check_system_requirements,
    generate_summary_report
)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run dimensionality reduction experiments on Fashion-MNIST'
    )
    
    parser.add_argument(
        '--subset-size', type=int, default=5000,
        help='Number of samples to use (default: 5000)'
    )
    
    parser.add_argument(
        '--methods', nargs='+', 
        default=['pca', 'lda', 'ica', 'tsne', 'umap', 'kernel_pca'],
        help='Methods to run (default: pca lda ica tsne umap kernel_pca)'
    )
    
    parser.add_argument(
        '--n-components', type=int, default=2,
        help='Number of components for dimensionality reduction (default: 2)'
    )
    
    parser.add_argument(
        '--output-dir', default='results/',
        help='Output directory for results (default: results/)'
    )
    
    parser.add_argument(
        '--preprocessing', choices=['standard', 'minmax', 'none'], 
        default='standard',
        help='Data preprocessing method (default: standard)'
    )
    
    parser.add_argument(
        '--skip-slow', action='store_true',
        help='Skip slow methods like t-SNE for large datasets'
    )
    
    parser.add_argument(
        '--random-state', type=int, default=42,
        help='Random state for reproducibility (default: 42)'
    )
    
    parser.add_argument(
        '--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    return parser.parse_args()

def filter_methods_by_feasibility(methods, n_samples, skip_slow):
    """Filter methods based on dataset size and computational constraints."""
    if skip_slow and n_samples > 10000:
        slow_methods = ['tsne', 'isomap', 'lle', 'autoencoder']
        methods = [m for m in methods if m not in slow_methods]
        logging.info(f"Skipping slow methods for large dataset: {slow_methods}")
    
    return methods

def setup_method_parameters():
    """Define method-specific parameters for optimal performance."""
    return {
        'tsne': {
            'perplexity': min(30, 50),  # Will be adjusted based on sample size
            'n_iter': 1000,
            'learning_rate': 200.0
        },
        'umap': {
            'n_neighbors': 15,
            'min_dist': 0.1,
            'metric': 'euclidean'
        },
        'kernel_pca': {
            'kernel': 'rbf',
            'gamma': 0.01
        },
        'isomap': {
            'n_neighbors': 10
        },
        'lle': {
            'n_neighbors': 10,
            'method': 'standard'
        },
        'autoencoder': {
            'hidden_layers': [128, 64],
            'epochs': 50,
            'batch_size': 256,
            'learning_rate': 0.001
        }
    }
