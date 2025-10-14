import unittest
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dimensionality_reducer import DimensionalityReducer, AutoEncoder
from data_loader import load_fashion_mnist


class TestDimensionalityReducer(unittest.TestCase):
    """Test cases for DimensionalityReducer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.reducer = DimensionalityReducer(random_state=42)
        # Load small sample for testing
        self.X_train, _, self.y_train, _ = load_fashion_mnist(subset_size=200)
    
    def test_pca_basic(self):
        """Test basic PCA functionality."""
        result = self.reducer.fit_transform_method(
            self.X_train, method_name='pca', n_components=2
        )
        
        self.assertNotIn('error', result)
        self.assertEqual(result['embedding'].shape, (200, 2))
        self.assertGreater(result['fit_time'], 0)
        self.assertIsNotNone(result['explained_variance_ratio'])
    
    def test_lda_with_labels(self):
        """Test LDA with labels."""
        result = self.reducer.fit_transform_method(
            self.X_train, self.y_train, method_name='lda', n_components=2
        )
        
        self.assertNotIn('error', result)
        self.assertEqual(result['embedding'].shape[1], 2)
        self.assertGreater(result['fit_time'], 0)
    
    def test_tsne_basic(self):
        """Test t-SNE functionality."""
        result = self.reducer.fit_transform_method(
            self.X_train, method_name='tsne', n_components=2, perplexity=10
        )
        
        self.assertNotIn('error', result)
        self.assertEqual(result['embedding'].shape, (200, 2))
        self.assertGreater(result['fit_time'], 0)
    
    def test_compare_methods(self):
        """Test comparing multiple methods."""
        methods = ['pca', 'ica']
        results = self.reducer.compare_methods(
            self.X_train, self.y_train, methods=methods, n_components=2
        )
        
        self.assertEqual(len(results), len(methods))
        for method in methods:
            self.assertIn(method, results)
            if 'error' not in results[method]:
                self.assertEqual(results[method]['embedding'].shape[1], 2)
    
    def test_method_info(self):
        """Test method information retrieval."""
        info = self.reducer.get_method_info()
        
        self.assertIn('pca', info)
        self.assertIn('tsne', info)
        self.assertEqual(info['pca']['type'], 'linear')
        self.assertEqual(info['tsne']['type'], 'nonlinear')


class TestAutoEncoder(unittest.TestCase):
    """Test cases for AutoEncoder class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.X_train, _, _, _ = load_fashion_mnist(subset_size=100)
    
    def test_autoencoder_basic(self):
        """Test basic autoencoder functionality."""
        autoencoder = AutoEncoder(
            encoding_dim=2, 
            hidden_layers=[32], 
            epochs=2,  # Quick test
            verbose=0
        )
        
        X_encoded = autoencoder.fit_transform(self.X_train)
        
        self.assertEqual(X_encoded.shape, (100, 2))
        self.assertIsNotNone(autoencoder.encoder)
        self.assertIsNotNone(autoencoder.autoencoder)
    
    def test_autoencoder_inverse_transform(self):
        """Test autoencoder reconstruction."""
        autoencoder = AutoEncoder(
            encoding_dim=10, 
            epochs=1,  # Very quick
            verbose=0
        )
        
        X_encoded = autoencoder.fit_transform(self.X_train[:50])
        X_reconstructed = autoencoder.inverse_transform(X_encoded)
        
        self.assertEqual(X_reconstructed.shape, (50, 784))


if __name__ == '__main__':
    unittest.main()