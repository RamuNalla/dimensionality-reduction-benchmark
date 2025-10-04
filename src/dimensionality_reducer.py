import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD, FastICA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding
import umap
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
import time
import logging
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
logger = logging.getLogger(__name__)

class AutoEncoder:
    
    def __init__(self, encoding_dim=2, hidden_layers=[128, 64], activation='relu', 
                 optimizer='adam', learning_rate=0.001, epochs=50, batch_size=256, verbose=0):
        """
        Initialize Autoencoder.
        
        Args:
            encoding_dim (int): Dimension of encoded representation
            hidden_layers (list): List of hidden layer sizes
            activation (str): Activation function
            optimizer (str): Optimizer name
            learning_rate (float): Learning rate
            epochs (int): Training epochs
            batch_size (int): Batch size
            verbose (int): Verbosity level
        """
        self.encoding_dim = encoding_dim
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.autoencoder = None
        self.encoder = None

    def fit(self, X):           # Train the autoencoder
        
        input_dim = X.shape[1]
        
        # Input layer
        input_layer = Input(shape=(input_dim,))
        
        # Encoder layers
        encoded = input_layer
        for hidden_size in self.hidden_layers:
            encoded = Dense(hidden_size, activation=self.activation)(encoded)
        
        # Bottleneck layer
        encoded = Dense(self.encoding_dim, activation=self.activation, name='encoded')(encoded)
        
        # Decoder layers
        decoded = encoded
        for hidden_size in reversed(self.hidden_layers):
            decoded = Dense(hidden_size, activation=self.activation)(decoded)
        
        # Output layer
        decoded = Dense(input_dim, activation='sigmoid')(decoded)
        
        # Create models
        self.autoencoder = Model(input_layer, decoded)
        self.encoder = Model(input_layer, encoded)
        
        # Compile
        if self.optimizer == 'adam':
            opt = Adam(learning_rate=self.learning_rate)
        else:
            opt = self.optimizer
            
        self.autoencoder.compile(optimizer=opt, loss='mse')
        
        # Train
        self.autoencoder.fit(
            X, X,
            epochs=self.epochs,
            batch_size=self.batch_size,
            shuffle=True,
            verbose=self.verbose
        )

    def transform(self, X):     # Encode data to lower dimension.
        """
        Args:
            X (np.ndarray): Input data 
        Returns:
            np.ndarray: Encoded data
        """
        if self.encoder is None:
            raise ValueError("Autoencoder must be fitted before transform")
        return self.encoder.predict(X, verbose=0)
    
    def fit_transform(self, X):    # Fit and transform data (convenience method that combines fit and transform).
        """
        Args:
            X (np.ndarray): Input data
            
        Returns:
            np.ndarray: Encoded data
        """
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X_encoded):     # Decode data from lower dimension.
        """
        Args:
            X_encoded (np.ndarray): Encoded data
            
        Returns:
            np.ndarray: Decoded data
        """
        if self.autoencoder is None:
            raise ValueError("Autoencoder must be fitted before inverse_transform")
        return self.autoencoder.predict(X_encoded, verbose=0)
    

class DimensionalityReducer:        # Unified interface for various dimensionality reduction techniques.

    def __init__(self, random_state=42):
        
        self.random_state = random_state
        self.methods = {}
        self.fitted_models = {}
        self.results = {}

    def _get_method(self, method_name, n_components=2, **kwargs):
        """
        Args:
            method_name (str): Name of the method
            n_components (int): Number of components
            **kwargs: Additional method-specific parameters
            
        Returns:
            sklearn estimator: Configured method instance
        """
        if method_name == 'pca':
            return PCA(n_components=n_components, random_state=self.random_state, **kwargs)
        
        elif method_name == 'lda':
            # LDA is limited by number of classes - 1
            return LDA(n_components=min(n_components, 9), **kwargs)
        
        elif method_name == 'ica':
            return FastICA(n_components=n_components, random_state=self.random_state, 
                          max_iter=1000, **kwargs)
        
        elif method_name == 'svd':
            return TruncatedSVD(n_components=n_components, random_state=self.random_state, **kwargs)
        
        elif method_name == 'kernel_pca':
            return KernelPCA(n_components=n_components, kernel='rbf', 
                           random_state=self.random_state, **kwargs)
        
        elif method_name == 'tsne':
            return TSNE(n_components=n_components, random_state=self.random_state,
                       perplexity=30, n_iter=1000, **kwargs)
        
        elif method_name == 'umap':
            return umap.UMAP(n_components=n_components, random_state=self.random_state,
                            n_neighbors=15, min_dist=0.1, **kwargs)
        
        elif method_name == 'isomap':
            return Isomap(n_components=n_components, n_neighbors=10, **kwargs)
        
        elif method_name == 'lle':
            return LocallyLinearEmbedding(n_components=n_components, 
                                        random_state=self.random_state,
                                        n_neighbors=10, **kwargs)
        
        elif method_name == 'autoencoder':
            return AutoEncoder(encoding_dim=n_components, **kwargs)
        
        else:
            raise ValueError(f"Unknown method: {method_name}")
        

    def fit_transform_method(self, X, y=None, method_name='pca', n_components=2, **kwargs): # Apply single dimensionality reduction method.
        """
        Args:
            X (np.ndarray): Input data
            y (np.ndarray, optional): Labels (required for LDA)
            method_name (str): Method to use
            n_components (int): Number of components
            **kwargs: Method-specific parameters
            
        Returns:
            dict: Results containing transformed data and metadata
        """
        logger.info(f"Applying {method_name.upper()} with {n_components} components...")
        
        # Get method instance
        method = self._get_method(method_name, n_components, **kwargs)
        
        # Record start time
        start_time = time.time()
        
        # Fit and transform
        try:
            if method_name == 'lda' and y is not None:
                X_transformed = method.fit_transform(X, y)
            else:
                X_transformed = method.fit_transform(X)
            
            fit_time = time.time() - start_time
            
            # Store fitted model
            self.fitted_models[method_name] = method
            
            # Calculate explained variance if available
            explained_variance_ratio = None
            if hasattr(method, 'explained_variance_ratio_'):
                explained_variance_ratio = method.explained_variance_ratio_
            
            result = {
                'method': method_name,
                'n_components': n_components,
                'embedding': X_transformed,
                'fit_time': fit_time,
                'explained_variance_ratio': explained_variance_ratio,
                'model': method,
                'parameters': kwargs
            }
            
            logger.info(f"{method_name.upper()} completed in {fit_time:.2f} seconds")
            return result
            
        except Exception as e:
            logger.error(f"Error in {method_name}: {str(e)}")
            return {
                'method': method_name,
                'error': str(e),
                'fit_time': time.time() - start_time
            }

    def compare_methods(self, X, y=None, methods=None, n_components=2, **method_kwargs):
        """
        Compare multiple dimensionality reduction methods.
        
        Args:
            X (np.ndarray): Input data
            y (np.ndarray, optional): Labels
            methods (list): List of method names to compare
            n_components (int): Number of components
            **method_kwargs: Method-specific parameters (nested dict)
            
        Returns:
            dict: Results for each method
        """
        if methods is None:
            methods = ['pca', 'lda', 'ica', 'svd', 'tsne', 'umap', 'kernel_pca', 'isomap', 'lle']
        
        results = {}
        
        for method_name in methods:
            # Get method-specific kwargs
            kwargs = method_kwargs.get(method_name, {})
            
            # Skip LDA if no labels provided
            if method_name == 'lda' and y is None:
                logger.warning("Skipping LDA: requires labels")
                continue
            
            try:
                result = self.fit_transform_method(
                    X, y, method_name, n_components, **kwargs
                )
                results[method_name] = result
                
            except Exception as e:
                logger.error(f"Failed to apply {method_name}: {str(e)}")
                results[method_name] = {'method': method_name, 'error': str(e)}
        
        self.results = results
        return results
    
    def transform_new_data(self, X_new, method_name):
        """
        Transform new data using fitted model.
        
        Args:
            X_new (np.ndarray): New data to transform
            method_name (str): Method to use
            
        Returns:
            np.ndarray: Transformed data
        """
        if method_name not in self.fitted_models:
            raise ValueError(f"Method {method_name} not fitted yet")
        
        method = self.fitted_models[method_name]
        
        # Handle methods that don't support transform on new data
        if method_name in ['tsne']:
            raise ValueError(f"{method_name} doesn't support transforming new data")
        
        try:
            return method.transform(X_new)
        except AttributeError:
            raise ValueError(f"{method_name} doesn't support transforming new data")