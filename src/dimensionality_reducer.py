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
    
    def fit_transform(self, X):    # Fit and transform data.
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