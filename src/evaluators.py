import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, silhouette_score, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr
import warnings
import logging

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class DimensionalityReductionEvaluator:     # Comprehensive evaluation of dimensionality reduction techniques.

    def __init__(self, random_state=42):    # Initialize evaluator.
        
        self.random_state = random_state
        self.metrics = {}

    def reconstruction_error(self, X_original, X_reduced, method_obj=None):     # Calculate reconstruction error between original and reconstructed data.
        """
        Calculate reconstruction error between original and reconstructed data.
        
        Args:
            X_original (np.ndarray): Original high-dimensional data
            X_reduced (np.ndarray): Reduced dimensional data
            method_obj: Fitted dimensionality reduction object
            
        Returns:
            float: Mean squared reconstruction error
        """
        try:
            if hasattr(method_obj, 'inverse_transform'):        # If inverse_transform method exists, use this method to convert the reduced data to the original high-dimensional data
                X_reconstructed = method_obj.inverse_transform(X_reduced)
            elif hasattr(method_obj, 'components_'):        # If not, then manually construct the input from low-dim data
                # For PCA-like methods
                X_reconstructed = X_reduced @ method_obj.components_ + method_obj.mean_
            else:
                # Cannot compute reconstruction error
                return np.nan
                
            mse = np.mean((X_original - X_reconstructed) ** 2)
            return mse
            
        except Exception as e:
            logger.warning(f"Could not compute reconstruction error: {e}")
            return np.nan
        
    def explained_variance_ratio(self, method_obj):
        """
        Get explained variance ratio if available.
        
        Args:
            method_obj: Fitted dimensionality reduction object
            
        Returns:
            float: Total explained variance ratio
        """
        if hasattr(method_obj, 'explained_variance_ratio_'):
            return np.sum(method_obj.explained_variance_ratio_)
        else:
            return np.nan
        
    
    def classification_performance(self, X_train_reduced, X_test_reduced, y_train, y_test, 
                                 classifiers=None, cv_folds=3):
        """
        Evaluate classification performance on reduced dimensional data.
        
        Args:
            X_train_reduced (np.ndarray): Training data in reduced dimensions
            X_test_reduced (np.ndarray): Test data in reduced dimensions  
            y_train (np.ndarray): Training labels
            y_test (np.ndarray): Test labels
            classifiers (dict): Dictionary of classifiers to test
            cv_folds (int): Cross-validation folds
            
        Returns:
            dict: Classification performance metrics
        """
        if classifiers is None:
            classifiers = {
                'knn': KNeighborsClassifier(n_neighbors=5),
                'logistic': LogisticRegression(random_state=self.random_state, max_iter=1000)
            }
        
        results = {}
        
        for clf_name, clf in classifiers.items():
            try:
                clf.fit(X_train_reduced, y_train)       # Fit classifier
                
                # Predictions
                y_pred = clf.predict(X_test_reduced)
                test_accuracy = accuracy_score(y_test, y_pred)
                
                cv_scores = cross_val_score(clf, X_train_reduced, y_train,  # Cross-validation on training set
                                          cv=cv_folds, scoring='accuracy')
                
                results[clf_name] = {
                    'test_accuracy': test_accuracy,
                    'cv_mean': np.mean(cv_scores),
                    'cv_std': np.std(cv_scores)
                }
                
            except Exception as e:
                logger.warning(f"Classification failed for {clf_name}: {e}")
                results[clf_name] = {'error': str(e)}
        
        return results 
    
    def clustering_performance(self, X_reduced, y_true=None, n_clusters=None):
        """
        Evaluate clustering performance on reduced data.
        
        Args:
            X_reduced (np.ndarray): Reduced dimensional data
            y_true (np.ndarray, optional): True labels for supervised evaluation
            n_clusters (int, optional): Number of clusters
            
        Returns:
            dict: Clustering performance metrics
        """
        if n_clusters is None and y_true is not None:
            n_clusters = len(np.unique(y_true))
        elif n_clusters is None:
            n_clusters = 10  # Default
        
        results = {}
        
        try:
            # Silhouette score (works without true labels)
            if X_reduced.shape[0] > 1 and X_reduced.shape[1] > 0:
                silhouette_avg = silhouette_score(X_reduced, 
                                                KMeans(n_clusters=n_clusters, 
                                                      random_state=self.random_state).fit_predict(X_reduced))
                results['silhouette_score'] = silhouette_avg
            else:
                results['silhouette_score'] = np.nan
                
        except Exception as e:
            logger.warning(f"Silhouette score calculation failed: {e}")
            results['silhouette_score'] = np.nan
        
        return results
