import unittest
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_loader import DataLoader, load_fashion_mnist

class TestDataLoader(unittest.TestCase):
    """Test cases for DataLoader class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.loader = DataLoader()
    
    def test_load_fashion_mnist_basic(self):
        """Test basic Fashion-MNIST loading."""
        X_train, X_test, y_train, y_test = load_fashion_mnist(subset_size=100)
        
        # Check shapes
        self.assertEqual(X_train.shape[0], 100)
        self.assertEqual(X_train.shape[1], 784)  # 28*28
        self.assertEqual(len(y_train), 100)
        self.assertGreater(len(X_test), 0)
        
        # Check data types
        self.assertEqual(X_train.dtype, np.float32)
        self.assertEqual(y_train.dtype, np.uint8)
        
        # Check value ranges (after normalization)
        self.assertGreaterEqual(X_train.min(), -5)  # Reasonable range after standardization
        self.assertLessEqual(X_train.max(), 5)
    
    def test_preprocessing_methods(self):
        """Test different preprocessing methods."""
        X_train_raw, _, _, _ = load_fashion_mnist(subset_size=50, preprocessing='none')
        X_train_std, _, _, _ = load_fashion_mnist(subset_size=50, preprocessing='standard')
        X_train_minmax, _, _, _ = load_fashion_mnist(subset_size=50, preprocessing='minmax')
        
        # Raw data should be in [0, 1]
        self.assertGreaterEqual(X_train_raw.min(), 0)
        self.assertLessEqual(X_train_raw.max(), 1)
        
        # Standard scaling should have ~0 mean
        self.assertAlmostEqual(X_train_std.mean(), 0, places=1)
        
        # MinMax should be in [0, 1]
        self.assertGreaterEqual(X_train_minmax.min(), 0)
        self.assertLessEqual(X_train_minmax.max(), 1)
    
    def test_class_info(self):
        """Test class information retrieval."""
        class_info = self.loader.get_class_info()
        
        self.assertEqual(len(class_info), 10)
        self.assertIn(0, class_info)
        self.assertIn(9, class_info)
        self.assertEqual(class_info[0], 'T-shirt/top')
    
    def test_sample_dataset(self):
        """Test sample dataset creation."""
        X_sample, y_sample = self.loader.create_sample_dataset(n_samples=100, n_classes=5)
        
        self.assertEqual(len(X_sample), 100)
        self.assertEqual(len(y_sample), 100)
        self.assertLessEqual(len(np.unique(y_sample)), 5)

if __name__ == '__main__':
    unittest.main()