import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import fashion_mnist
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Fashion-MNIST class labels
FASHION_MNIST_LABELS = {
    0: 'T-shirt/top',
    1: 'Trouser',
    2: 'Pullover', 
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot'
}

class DataLoader:

    def __init__(self, data_dir='data/'):      # Initialize DataLoader 
        
        self.data_dir = data_dir
        self.scaler = None
        os.makedirs(data_dir, exist_ok=True)

    def load_fashion_mnist(self, subset_size=None, random_state=42):        # Load Fashion-MNIST dataset from Keras.
        """
        Args:
            subset_size (int, optional): Use only a subset of data for faster experimentation
        """
        logger.info("Loading Fashion-MNIST dataset...")
        
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()    # Load data from Keras
        
        # Reshape to flatten images
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)
        
        # Convert to float32 and normalize to [0, 1]
        X_train = X_train.astype('float32') / 255.0
        X_test = X_test.astype('float32') / 255.0
        
        # Use subset if specified
        if subset_size:                 # Use only a subset of data for faster experimentation  
            indices = np.random.RandomState(random_state).choice(
                len(X_train), size=min(subset_size, len(X_train)), replace=False
            )
            X_train = X_train[indices]
            y_train = y_train[indices]
            
            test_indices = np.random.RandomState(random_state).choice(
                len(X_test), size=min(subset_size//5, len(X_test)), replace=False
            )
            X_test = X_test[test_indices]
            y_test = y_test[test_indices]
        
        logger.info(f"Loaded {X_train.shape[0]} training and {X_test.shape[0]} test samples")
        logger.info(f"Feature dimension: {X_train.shape[1]}")


    def preprocess_data(self, X_train, X_test, method='standard'):      # Preprocess the data using various scaling methods.
        """
        Args: 
            method (str): Scaling method ('standard', 'minmax', 'none')
        """
        logger.info(f"Preprocessing data with {method} scaling...")
        
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        elif method == 'none':
            return X_train, X_test
        else:
            raise ValueError(f"Unknown preprocessing method: {method}")
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled
    
    
    def get_class_info(self):           # Get information about Fashion-MNIST classes.
        return FASHION_MNIST_LABELS
    

    def create_sample_dataset(self, n_samples=1000, n_classes=10):  # Create a smaller balanced sample for quick testing.
        """
        Args:
            n_samples (int): Total number of samples
            n_classes (int): Number of classes to include
        """
        X_train, _, y_train, _ = self.load_fashion_mnist()
        
        samples_per_class = n_samples // n_classes              # Select balanced samples from each class
        selected_indices = []
        
        for class_id in range(n_classes):
            class_indices = np.where(y_train == class_id)[0]
            selected = np.random.choice(
                class_indices, 
                size=min(samples_per_class, len(class_indices)), 
                replace=False
            )
            selected_indices.extend(selected)
        
        selected_indices = np.array(selected_indices)
        np.random.shuffle(selected_indices)
        
        return X_train[selected_indices], y_train[selected_indices]
    

    def save_processed_data(self, X_train, X_test, y_train, y_test, filename_prefix='processed'):   # Save processed data
       
        filepath = os.path.join(self.data_dir, f'{filename_prefix}')
        
        np.savez_compressed(
            filepath,
            X_train=X_train,
            X_test=X_test, 
            y_train=y_train,
            y_test=y_test
        )
        logger.info(f"Saved processed data to {filepath}.npz")

    
    def load_processed_data(self, filename_prefix='processed'):     # Load processed data from disk.
        
        filepath = os.path.join(self.data_dir, f'{filename_prefix}.npz')
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Processed data not found at {filepath}")
        
        data = np.load(filepath)
        logger.info(f"Loaded processed data from {filepath}")
        
        return data['X_train'], data['X_test'], data['y_train'], data['y_test']
    
    
    