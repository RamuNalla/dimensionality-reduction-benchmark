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

def run_experiments(args):
    """Run all dimensionality reduction experiments."""
    
    # Setup
    create_directory_structure(args.output_dir)
    log_file = os.path.join(args.output_dir, 'experiment.log')
    setup_logging(getattr(logging, args.log_level), log_file)
    
    logger = logging.getLogger(__name__)
    logger.info("Starting dimensionality reduction experiments")
    logger.info(f"Arguments: {vars(args)}")
    
    # Initialize experiment logger
    exp_logger = ExperimentLogger(os.path.join(args.output_dir, 'logs'))
    
    try:
        # Load data
        with Timer("Data loading"):
            X_train, X_test, y_train, y_test = load_fashion_mnist(
                subset_size=args.subset_size,
                preprocessing=args.preprocessing,
                random_state=args.random_state
            )
        
        logger.info(f"Loaded {X_train.shape[0]} training and {X_test.shape[0]} test samples")
        logger.info(f"Feature dimension: {X_train.shape[1]}")
        
        # Check system requirements
        system_check = check_system_requirements(
            X_train.shape[0], X_train.shape[1], args.methods
        )
        logger.info(f"System feasibility check: {system_check['feasible']}")
        logger.info(f"Recommendation: {system_check['recommendation']}")
        
        if not system_check['feasible']:
            logger.warning("System may not have sufficient resources. Consider reducing data size.")
        
        # Filter methods
        methods = filter_methods_by_feasibility(
            args.methods, X_train.shape[0], args.skip_slow
        )
        logger.info(f"Methods to run: {methods}")
        
        # Setup method parameters
        method_params = setup_method_parameters()
        
        # Adjust t-SNE perplexity based on sample size
        if 'tsne' in methods and 'tsne' in method_params:
            max_perplexity = min(30, (X_train.shape[0] - 1) // 3)
            method_params['tsne']['perplexity'] = max_perplexity
            logger.info(f"Adjusted t-SNE perplexity to {max_perplexity}")
        
        # Initialize dimensionality reducer
        reducer = DimensionalityReducer(random_state=args.random_state)
        
        # Run dimensionality reduction
        with Timer("Dimensionality reduction"):
            results = reducer.compare_methods(
                X_train, y_train,
                methods=methods,
                n_components=args.n_components,
                **method_params
            )
        
        # Log individual method results
        for method_name, result in results.items():
            if 'error' not in result:
                exp_logger.log_experiment(
                    method_name=method_name,
                    parameters={
                        'n_components': args.n_components,
                        **method_params.get(method_name, {})
                    },
                    results={
                        'fit_time': result.get('fit_time'),
                        'embedding_shape': result['embedding'].shape if 'embedding' in result else None
                    }
                )
        
        # Evaluate results
        with Timer("Evaluation"):
            evaluator = evaluate_all_metrics(
                results, X_train, X_test, y_train, y_test
            )
        
        # Update experiment logs with metrics
        for method_name, evaluation in evaluator.metrics.items():
            if 'error' not in evaluation:
                # Find corresponding experiment and update with metrics
                for exp in exp_logger.experiments:
                    if exp['method'] == method_name:
                        exp['metrics'] = evaluation
                        break
        
        # Save results
        results_file = os.path.join(args.output_dir, 'results.pkl')
        evaluations_file = os.path.join(args.output_dir, 'evaluations.pkl')
        
        save_results(results, results_file, format='pickle')
        save_results(evaluator.metrics, evaluations_file, format='pickle')
        
        # Save comparison table
        comparison_table = evaluator.create_comparison_table(evaluator.metrics)
        comparison_file = os.path.join(args.output_dir, 'metrics', 'comparison_table.csv')
        os.makedirs(os.path.dirname(comparison_file), exist_ok=True)
        comparison_table.to_csv(comparison_file, index=False)
        
        logger.info("Comparison table:")
        logger.info(f"\n{comparison_table.to_string(index=False)}")
        
        # Generate visualizations
        with Timer("Visualization generation"):
            save_all_plots(
                results, evaluator.metrics, y_train,
                output_dir=os.path.join(args.output_dir, 'figures')
            )
        
        # Generate summary report
        report_file = os.path.join(args.output_dir, 'summary_report.txt')
        report_text = generate_summary_report(
            results, evaluator.metrics, output_path=report_file
        )
        
        logger.info("Summary report generated:")
        logger.info(f"\n{report_text}")
        
        # Find and log best methods
        best_exp = exp_logger.get_best_results(metric='silhouette_score')
        if best_exp:
            logger.info(f"Best method (by silhouette score): {best_exp['method']}")
        
        logger.info("Experiments completed successfully!")
        
        return {
            'results': results,
            'evaluations': evaluator.metrics,
            'comparison_table': comparison_table,
            'summary_report': report_text
        }
        
    except Exception as e:
        logger.error(f"Experiment failed: {str(e)}", exc_info=True)
        raise
    
    finally:
        # Clean up
        logging.shutdown()

