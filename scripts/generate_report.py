import os
import sys
import argparse
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils import load_results
from visualizers import DimensionalityReductionVisualizer, create_interactive_plot


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate interactive report from experiment results'
    )
    
    parser.add_argument(
        '--results-dir', default='results/',
        help='Directory containing experiment results (default: results/)'
    )
    
    parser.add_argument(
        '--output-file', default='interactive_report.html',
        help='Output HTML file name (default: interactive_report.html)'
    )
    
    parser.add_argument(
        '--include-3d', action='store_true',
        help='Include 3D visualizations'
    )
    
    return parser.parse_args()


def load_experiment_data(results_dir):
    """Load all experiment data."""
    results_file = os.path.join(results_dir, 'results.pkl')
    evaluations_file = os.path.join(results_dir, 'evaluations.pkl')
    
    if not os.path.exists(results_file):
        raise FileNotFoundError(f"Results file not found: {results_file}")
    
    if not os.path.exists(evaluations_file):
        raise FileNotFoundError(f"Evaluations file not found: {evaluations_file}")
    
    results = load_results(results_file)
    evaluations = load_results(evaluations_file)
    
    return results, evaluations