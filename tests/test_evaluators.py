import unittest
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from evaluators import DimensionalityReductionEvaluator, evaluate_all_metrics
from dimensionality_reducer import DimensionalityReducer
from data_loader import load_fashion_mnist

class TestDimensionalityReductionEvaluator(unittest.TestCase):
    """Test cases for evaluation functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.evaluator = DimensionalityReductionEvaluator(random_state=42)
        # Load small sample for testing
        self.X_train, self.X_test, self.y_train, self.y_test = load_fashion_mnist(subset_size=100)
        
        # Create sample results
        self.reducer = DimensionalityReducer(random_state=42)
        self.results = self.reducer.compare_methods(
            self.X_train, self.y_train, 
            methods=['pca'], 
            n_components=2
        )
    
    def test_clustering_performance(self):
        """Test clustering performance evaluation."""
        X_reduced = self.results['pca']['embedding']
        clustering_metrics = self.evaluator.clustering_performance(X_reduced, self.y_train)
        
        self.assertIn('silhouette_score', clustering_metrics)
        self.assertIsInstance(clustering_metrics['silhouette_score'], (int, float))
        self.assertGreaterEqual(clustering_metrics['silhouette_score'], -1)
        self.assertLessEqual(clustering_metrics['silhouette_score'], 1)
    
    def test_classification_performance(self):
        """Test classification performance evaluation."""
        X_train_reduced = self.results['pca']['embedding']
        # Simple test with same data
        clf_metrics = self.evaluator.classification_performance(
            X_train_reduced, X_train_reduced, self.y_train, self.y_train
        )
        
        self.assertIn('knn', clf_metrics)
        self.assertIn('logistic', clf_metrics)
        
        if 'error' not in clf_metrics['knn']:
            self.assertIn('test_accuracy', clf_metrics['knn'])
            self.assertGreaterEqual(clf_metrics['knn']['test_accuracy'], 0)
            self.assertLessEqual(clf_metrics['knn']['test_accuracy'], 1)
    
    def test_neighborhood_preservation(self):
        """Test neighborhood preservation metrics."""
        X_reduced = self.results['pca']['embedding']
        neighborhood_metrics = self.evaluator.neighborhood_preservation(
            self.X_train[:50], X_reduced[:50]  # Small sample for speed
        )
        
        self.assertIn('trustworthiness', neighborhood_metrics)
        self.assertIn('continuity', neighborhood_metrics)
        
        for metric in ['trustworthiness', 'continuity']:
            value = neighborhood_metrics[metric]
            if not np.isnan(value):
                self.assertGreaterEqual(value, 0)
                self.assertLessEqual(value, 1)
    
    def test_evaluate_single_method(self):
        """Test single method evaluation."""
        evaluation = self.evaluator.evaluate_single_method(
            self.results['pca'], self.X_train, self.X_test, self.y_train, self.y_test
        )
        
        self.assertEqual(evaluation['method'], 'pca')
        self.assertIn('silhouette_score', evaluation)
        self.assertIn('trustworthiness', evaluation)
        self.assertIn('fit_time', evaluation)
    
    def test_create_comparison_table(self):
        """Test comparison table creation."""
        evaluations = evaluate_all_metrics(
            self.results, self.X_train, self.X_test, self.y_train, self.y_test
        )
        
        table = evaluations.create_comparison_table(evaluations.metrics)
        
        self.assertGreater(len(table), 0)
        self.assertIn('Method', table.columns)
        self.assertIn('Fit Time (s)', table.columns)
        self.assertIn('Silhouette Score', table.columns)


if __name__ == '__main__':
    unittest.main()