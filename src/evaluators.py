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

    def neighborhood_preservation(self, X_original, X_reduced, k=10):
        """
        Evaluate how well local neighborhoods are preserved.
        
        Args:
            X_original (np.ndarray): Original high-dimensional data
            X_reduced (np.ndarray): Reduced dimensional data
            k (int): Number of neighbors to consider
            
        Returns:
            dict: Neighborhood preservation metrics
        """
        try:
            n_samples = min(X_original.shape[0], 1000)  # Limit for computational efficiency
            indices = np.random.choice(X_original.shape[0], n_samples, replace=False)
            
            X_orig_sample = X_original[indices]
            X_red_sample = X_reduced[indices]
            
            # Calculate pairwise distances
            dist_orig = squareform(pdist(X_orig_sample))
            dist_red = squareform(pdist(X_red_sample))
            
            # Find k nearest neighbors in both spaces
            trustworthiness_scores = []
            continuity_scores = []
            
            for i in range(n_samples):
                # Original space neighbors
                orig_neighbors = np.argsort(dist_orig[i])[1:k+1]
                
                # Reduced space neighbors  
                red_neighbors = np.argsort(dist_red[i])[1:k+1]
                
                # Trustworthiness: do close points in reduced space stay close in original?
                trust = len(np.intersect1d(orig_neighbors, red_neighbors)) / k
                trustworthiness_scores.append(trust)
                
                # Continuity: do close points in original space stay close in reduced?
                cont = len(np.intersect1d(red_neighbors, orig_neighbors)) / k
                continuity_scores.append(cont)
            
            results = {
                'trustworthiness': np.mean(trustworthiness_scores),
                'continuity': np.mean(continuity_scores)
            }
            
        except Exception as e:
            logger.warning(f"Neighborhood preservation calculation failed: {e}")
            results = {
                'trustworthiness': np.nan,
                'continuity': np.nan
            }
        
        return results
    
    def distance_correlation(self, X_original, X_reduced, method='pearson'):
        """
        Compute correlation between pairwise distances in original and reduced space.
        
        Args:
            X_original (np.ndarray): Original high-dimensional data
            X_reduced (np.ndarray): Reduced dimensional data
            method (str): Correlation method ('pearson' or 'spearman')
            
        Returns:
            float: Distance correlation
        """
        try:
            # Sample for efficiency
            n_samples = min(X_original.shape[0], 500)
            indices = np.random.choice(X_original.shape[0], n_samples, replace=False)
            
            X_orig_sample = X_original[indices]
            X_red_sample = X_reduced[indices]
            
            # Calculate pairwise distances
            dist_orig = pdist(X_orig_sample)
            dist_red = pdist(X_red_sample)
            
            # Compute correlation
            if method == 'pearson':
                corr, _ = pearsonr(dist_orig, dist_red)
            elif method == 'spearman':
                corr, _ = spearmanr(dist_orig, dist_red)
            else:
                raise ValueError(f"Unknown correlation method: {method}")
                
            return corr
            
        except Exception as e:
            logger.warning(f"Distance correlation calculation failed: {e}")
            return np.nan

    def evaluate_single_method(self, result_dict, X_original, X_test_original=None, 
                              y_train=None, y_test=None):
        """
        Comprehensive evaluation of a single dimensionality reduction result.
        
        Args:
            result_dict (dict): Result from dimensionality reduction
            X_original (np.ndarray): Original training data
            X_test_original (np.ndarray, optional): Original test data
            y_train (np.ndarray, optional): Training labels
            y_test (np.ndarray, optional): Test labels
            
        Returns:
            dict: Comprehensive evaluation metrics
        """
        if 'error' in result_dict:
            return {'error': result_dict['error']}
        
        method_name = result_dict['method']
        X_reduced = result_dict['embedding']
        method_obj = result_dict.get('model')
        
        logger.info(f"Evaluating {method_name}...")
        
        evaluation = {
            'method': method_name,
            'n_components': result_dict.get('n_components', X_reduced.shape[1]),
            'fit_time': result_dict.get('fit_time', np.nan),
            'embedding_shape': X_reduced.shape
        }
        
        # Basic metrics
        evaluation['reconstruction_error'] = self.reconstruction_error(
            X_original, X_reduced, method_obj
        )
        evaluation['explained_variance_ratio'] = self.explained_variance_ratio(method_obj)
        
        # Distance preservation
        evaluation.update(self.neighborhood_preservation(X_original, X_reduced))
        evaluation['distance_correlation_pearson'] = self.distance_correlation(
            X_original, X_reduced, 'pearson'
        )
        evaluation['distance_correlation_spearman'] = self.distance_correlation(
            X_original, X_reduced, 'spearman'
        )
        
        # Clustering performance
        clustering_metrics = self.clustering_performance(X_reduced, y_train)
        evaluation.update(clustering_metrics)
        
        # Classification performance (if labels available)
        if y_train is not None:
            # Need to transform test data if available
            X_test_reduced = None
            if X_test_original is not None and hasattr(method_obj, 'transform'):
                try:
                    X_test_reduced = method_obj.transform(X_test_original)
                except:
                    pass
            
            if X_test_reduced is not None and y_test is not None:
                clf_metrics = self.classification_performance(
                    X_reduced, X_test_reduced, y_train, y_test
                )
                evaluation['classification'] = clf_metrics
        
        return evaluation

    def evaluate_all_methods(self, results_dict, X_original, X_test_original=None, 
                           y_train=None, y_test=None):
        """
        Evaluate all methods in results dictionary.
        
        Args:
            results_dict (dict): Dictionary of dimensionality reduction results
            X_original (np.ndarray): Original training data
            X_test_original (np.ndarray, optional): Original test data  
            y_train (np.ndarray, optional): Training labels
            y_test (np.ndarray, optional): Test labels
            
        Returns:
            dict: Evaluation results for all methods
        """
        all_evaluations = {}
        
        for method_name, result in results_dict.items():
            evaluation = self.evaluate_single_method(
                result, X_original, X_test_original, y_train, y_test
            )
            all_evaluations[method_name] = evaluation
        
        return all_evaluations
    
    def create_comparison_table(self, evaluations):
        """
        Create a comparison table of evaluation metrics.
        
        Args:
            evaluations (dict): Dictionary of evaluation results
            
        Returns:
            pd.DataFrame: Comparison table
        """
        rows = []
        
        for method_name, eval_dict in evaluations.items():
            if 'error' in eval_dict:
                continue
                
            row = {
                'Method': method_name,
                'Components': eval_dict.get('n_components', np.nan),
                'Fit Time (s)': eval_dict.get('fit_time', np.nan),
                'Reconstruction Error': eval_dict.get('reconstruction_error', np.nan),
                'Explained Variance': eval_dict.get('explained_variance_ratio', np.nan),
                'Silhouette Score': eval_dict.get('silhouette_score', np.nan),
                'Trustworthiness': eval_dict.get('trustworthiness', np.nan),
                'Continuity': eval_dict.get('continuity', np.nan),
                'Distance Correlation': eval_dict.get('distance_correlation_pearson', np.nan)
            }
            
            # Add classification metrics if available
            if 'classification' in eval_dict:
                clf_metrics = eval_dict['classification']
                if 'knn' in clf_metrics and 'error' not in clf_metrics['knn']:
                    row['KNN Accuracy'] = clf_metrics['knn'].get('test_accuracy', np.nan)
                if 'logistic' in clf_metrics and 'error' not in clf_metrics['logistic']:
                    row['Logistic Accuracy'] = clf_metrics['logistic'].get('test_accuracy', np.nan)
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        return df.round(4)
    

def evaluate_all_metrics(results_dict, X_original, X_test_original=None, 
                        y_train=None, y_test=None):
    """
    Convenience function to evaluate all methods with all metrics.
    
    Args:
        results_dict (dict): Dictionary of dimensionality reduction results
        X_original (np.ndarray): Original training data
        X_test_original (np.ndarray, optional): Original test data
        y_train (np.ndarray, optional): Training labels  
        y_test (np.ndarray, optional): Test labels
        
    Returns:
        DimensionalityReductionEvaluator: Evaluator with computed metrics
    """
    evaluator = DimensionalityReductionEvaluator()
    
    all_evaluations = evaluator.evaluate_all_methods(
        results_dict, X_original, X_test_original, y_train, y_test
    )
    
    evaluator.metrics = all_evaluations
    return evaluator


if __name__ == "__main__":
    # Example usage
    from src.data_loader import load_fashion_mnist
    from src.dimensionality_reducer import DimensionalityReducer
    
    # Load data
    X_train, X_test, y_train, y_test = load_fashion_mnist(subset_size=1000)
    
    # Apply dimensionality reduction
    reducer = DimensionalityReducer()
    results = reducer.compare_methods(
        X_train, y_train, 
        methods=['pca', 'tsne', 'umap'], 
        n_components=2
    )
    
    # Evaluate results
    evaluator = evaluate_all_metrics(results, X_train, X_test, y_train, y_test)
    
    # Create comparison table
    comparison_table = evaluator.create_comparison_table(evaluator.metrics)
    print("Evaluation Results:")
    print(comparison_table.to_string(index=False))
