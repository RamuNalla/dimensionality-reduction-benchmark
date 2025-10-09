import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from sklearn.metrics import confusion_matrix
import warnings
import logging

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

# Set style
plt.style.use('default')
sns.set_palette("husl")

# Fashion-MNIST class labels for visualization
FASHION_LABELS = {
    0: 'T-shirt/top', 1: 'Trouser', 2: 'Pullover', 3: 'Dress', 4: 'Coat',
    5: 'Sandal', 6: 'Shirt', 7: 'Sneaker', 8: 'Bag', 9: 'Ankle boot'
}

class DimensionalityReductionVisualizer:
    """
    Comprehensive visualization toolkit for dimensionality reduction results.
    """

    def __init__(self, figsize=(12, 8), color_palette='tab10'):
        """
        Initialize visualizer.
        
        Args:
            figsize (tuple): Default figure size
            color_palette (str): Color palette for plots
        """
        self.figsize = figsize
        self.color_palette = color_palette

    def plot_2d_embedding(self, X_embedded, y=None, title="2D Embedding", 
                         labels=None, save_path=None, interactive=True):
        """
        Create 2D scatter plot of embedded data.
        
        Args:
            X_embedded (np.ndarray): 2D embedded data
            y (np.ndarray, optional): Labels for coloring
            title (str): Plot title
            labels (dict, optional): Label mapping dictionary
            save_path (str, optional): Path to save plot
            interactive (bool): Create interactive plotly plot
            
        Returns:
            matplotlib.figure.Figure or plotly.graph_objects.Figure
        """
        if X_embedded.shape[1] != 2:
            raise ValueError("Input must be 2-dimensional")
        
        if interactive:
            return self._plot_2d_interactive(X_embedded, y, title, labels)
        else:
            return self._plot_2d_static(X_embedded, y, title, labels, save_path)

    def _plot_2d_interactive(self, X_embedded, y, title, labels):
        """Create interactive 2D plot using Plotly."""
        if labels is None:
            labels = FASHION_LABELS
        
        if y is not None:
            color_labels = [labels.get(label, f'Class {label}') for label in y]
            fig = px.scatter(
                x=X_embedded[:, 0], y=X_embedded[:, 1],
                color=color_labels,
                title=title,
                labels={'x': 'Component 1', 'y': 'Component 2'},
                opacity=0.7
            )
        else:
            fig = px.scatter(
                x=X_embedded[:, 0], y=X_embedded[:, 1],
                title=title,
                labels={'x': 'Component 1', 'y': 'Component 2'},
                opacity=0.7
            )
        
        fig.update_layout(
            width=800, height=600,
            title_x=0.5,
            showlegend=True if y is not None else False
        )
        return fig
    
    def _plot_2d_static(self, X_embedded, y, title, labels, save_path):
        """Create static 2D plot using Matplotlib."""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        if y is not None:
            scatter = ax.scatter(X_embedded[:, 0], X_embedded[:, 1], 
                               c=y, cmap='tab10', alpha=0.7, s=20)
            
            # Add legend if labels provided
            if labels:
                unique_labels = np.unique(y)
                legend_elements = []
                colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
                
                for i, label in enumerate(unique_labels):
                    legend_elements.append(
                        plt.Line2D([0], [0], marker='o', color='w', 
                                  markerfacecolor=colors[i], markersize=8,
                                  label=labels.get(label, f'Class {label}'))
                    )
                ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            ax.scatter(X_embedded[:, 0], X_embedded[:, 1], alpha=0.7, s=20)
        
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


    def plot_3d_embedding(self, X_embedded, y=None, title="3D Embedding", 
                         labels=None, interactive=True):
        """
        Create 3D scatter plot of embedded data.
        
        Args:
            X_embedded (np.ndarray): 3D embedded data
            y (np.ndarray, optional): Labels for coloring
            title (str): Plot title
            labels (dict, optional): Label mapping dictionary
            interactive (bool): Create interactive plotly plot
            
        Returns:
            plotly.graph_objects.Figure or matplotlib.figure.Figure
        """
        if X_embedded.shape[1] != 3:
            raise ValueError("Input must be 3-dimensional")
        
        if labels is None:
            labels = FASHION_LABELS
        
        if interactive:
            if y is not None:
                color_labels = [labels.get(label, f'Class {label}') for label in y]
                fig = px.scatter_3d(
                    x=X_embedded[:, 0], y=X_embedded[:, 1], z=X_embedded[:, 2],
                    color=color_labels,
                    title=title,
                    labels={'x': 'Component 1', 'y': 'Component 2', 'z': 'Component 3'},
                    opacity=0.7
                )
            else:
                fig = px.scatter_3d(
                    x=X_embedded[:, 0], y=X_embedded[:, 1], z=X_embedded[:, 2],
                    title=title,
                    labels={'x': 'Component 1', 'y': 'Component 2', 'z': 'Component 3'},
                    opacity=0.7
                )
            
            fig.update_layout(width=800, height=600, title_x=0.5)
            return fig
        
        else:
            fig = plt.figure(figsize=self.figsize)
            ax = fig.add_subplot(111, projection='3d')
            
            if y is not None:
                scatter = ax.scatter(X_embedded[:, 0], X_embedded[:, 1], X_embedded[:, 2],
                                   c=y, cmap='tab10', alpha=0.7, s=20)
                plt.colorbar(scatter)
            else:
                ax.scatter(X_embedded[:, 0], X_embedded[:, 1], X_embedded[:, 2],
                          alpha=0.7, s=20)
            
            ax.set_xlabel('Component 1')
            ax.set_ylabel('Component 2') 
            ax.set_zlabel('Component 3')
            ax.set_title(title)
            
            return fig
        
    
    def plot_multiple_embeddings(self, results_dict, y=None, ncols=3, 
                                save_path=None, interactive=False):
        """
        Plot multiple embeddings in a grid layout.
        
        Args:
            results_dict (dict): Dictionary of dimensionality reduction results
            y (np.ndarray, optional): Labels for coloring
            ncols (int): Number of columns in grid
            save_path (str, optional): Path to save plot
            interactive (bool): Create interactive plots
            
        Returns:
            matplotlib.figure.Figure or list of plotly figures
        """
        valid_results = {k: v for k, v in results_dict.items() 
                        if 'embedding' in v and v['embedding'].shape[1] >= 2}
        
        if not valid_results:
            logger.warning("No valid embeddings found for plotting")
            return None
        
        if interactive:
            return self._plot_multiple_interactive(valid_results, y)
        else:
            return self._plot_multiple_static(valid_results, y, ncols, save_path)
        
    
    def _plot_multiple_static(self, results_dict, y, ncols, save_path):
        """Create multiple static plots in grid."""
        n_methods = len(results_dict)
        nrows = (n_methods + ncols - 1) // ncols
        
        fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
        if nrows == 1 and ncols == 1:
            axes = [axes]
        elif nrows == 1:
            axes = axes
        else:
            axes = axes.flatten()
        
        for idx, (method_name, result) in enumerate(results_dict.items()):
            ax = axes[idx] if n_methods > 1 else axes[0]
            embedding = result['embedding']
            
            if embedding.shape[1] >= 2:
                if y is not None:
                    scatter = ax.scatter(embedding[:, 0], embedding[:, 1], 
                                       c=y, cmap='tab10', alpha=0.7, s=20)
                else:
                    ax.scatter(embedding[:, 0], embedding[:, 1], alpha=0.7, s=20)
                
                ax.set_title(f"{method_name.upper()}")
                ax.set_xlabel('Component 1')
                ax.set_ylabel('Component 2')
                ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(n_methods, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def _plot_multiple_interactive(self, results_dict, y):
        """Create multiple interactive plots."""
        figures = []
        
        for method_name, result in results_dict.items():
            embedding = result['embedding']
            
            if embedding.shape[1] >= 2:
                fig = self.plot_2d_embedding(
                    embedding[:, :2], y, 
                    title=f"{method_name.upper()} Embedding",
                    interactive=True
                )
                figures.append(fig)
        
        return figures
    
    def plot_explained_variance(self, results_dict, save_path=None):
        """
        Plot explained variance ratio for methods that support it.
        
        Args:
            results_dict (dict): Dictionary of dimensionality reduction results
            save_path (str, optional): Path to save plot
            
        Returns:
            matplotlib.figure.Figure
        """
        variance_data = []
        
        for method_name, result in results_dict.items():
            if 'explained_variance_ratio' in result and result['explained_variance_ratio'] is not None:
                variance_ratio = result['explained_variance_ratio']
                if hasattr(variance_ratio, '__len__'):
                    # Individual components
                    for i, var_ratio in enumerate(variance_ratio):
                        variance_data.append({
                            'Method': method_name,
                            'Component': f'PC{i+1}',
                            'Explained_Variance': var_ratio
                        })
                else:
                    # Total variance
                    variance_data.append({
                        'Method': method_name,
                        'Component': 'Total',
                        'Explained_Variance': variance_ratio
                    })
        
        if not variance_data:
            logger.warning("No explained variance data found")
            return None
        
        df = pd.DataFrame(variance_data)
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        if 'Component' in df.columns and df['Component'].nunique() > 1:
            sns.barplot(data=df, x='Method', y='Explained_Variance', hue='Component', ax=ax)
        else:
            sns.barplot(data=df, x='Method', y='Explained_Variance', ax=ax)
        
        ax.set_title('Explained Variance Ratio by Method')
        ax.set_ylabel('Explained Variance Ratio')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_performance_comparison(self, evaluations, metrics=None, save_path=None):
        """
        Create comparison plot of performance metrics.
        
        Args:
            evaluations (dict): Dictionary of evaluation results
            metrics (list, optional): List of metrics to plot
            save_path (str, optional): Path to save plot
            
        Returns:
            matplotlib.figure.Figure
        """
        if metrics is None:
            metrics = ['silhouette_score', 'trustworthiness', 'continuity', 
                      'distance_correlation_pearson']
        
        # Prepare data
        plot_data = []
        for method, eval_dict in evaluations.items():
            if 'error' in eval_dict:
                continue
                
            for metric in metrics:
                if metric in eval_dict and not pd.isna(eval_dict[metric]):
                    plot_data.append({
                        'Method': method,
                        'Metric': metric,
                        'Value': eval_dict[metric]
                    })
        
        if not plot_data:
            logger.warning("No valid performance data found")
            return None
        
        df = pd.DataFrame(plot_data)
        
        # Create subplots for each metric
        n_metrics = len(metrics)
        ncols = min(2, n_metrics)
        nrows = (n_metrics + ncols - 1) // ncols
        
        fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 4*nrows))
        if n_metrics == 1:
            axes = [axes]
        elif nrows == 1:
            axes = axes
        else:
            axes = axes.flatten()
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx] if n_metrics > 1 else axes[0]
            metric_data = df[df['Metric'] == metric]
            
            if not metric_data.empty:
                sns.barplot(data=metric_data, x='Method', y='Value', ax=ax)
                ax.set_title(f'{metric.replace("_", " ").title()}')
                ax.set_ylabel('Score')
                plt.sca(ax)
                plt.xticks(rotation=45)
        
        # Hide unused subplots
        for idx in range(n_metrics, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    

    def plot_runtime_comparison(self, results_dict, save_path=None):
        """
        Plot runtime comparison across methods.
        
        Args:
            results_dict (dict): Dictionary of dimensionality reduction results
            save_path (str, optional): Path to save plot
            
        Returns:
            matplotlib.figure.Figure
        """
        runtime_data = []
        
        for method_name, result in results_dict.items():
            if 'fit_time' in result and result['fit_time'] is not None:
                runtime_data.append({
                    'Method': method_name,
                    'Runtime_seconds': result['fit_time']
                })
        
        if not runtime_data:
            logger.warning("No runtime data found")
            return None
        
        df = pd.DataFrame(runtime_data).sort_values('Runtime_seconds')
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(df['Method'], df['Runtime_seconds'])
        
        # Color bars by runtime (green=fast, red=slow)
        colors = plt.cm.RdYlGn_r(df['Runtime_seconds'] / df['Runtime_seconds'].max())
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        ax.set_title('Runtime Comparison Across Methods')
        ax.set_ylabel('Runtime (seconds)')
        ax.set_xlabel('Method')
        plt.xticks(rotation=45)
        plt.yscale('log')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}s', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_interactive_dashboard(self, results_dict, evaluations, y=None):
        """
        Create interactive dashboard with multiple visualizations.
        
        Args:
            results_dict (dict): Dictionary of dimensionality reduction results
            evaluations (dict): Dictionary of evaluation results  
            y (np.ndarray, optional): Labels for coloring
            
        Returns:
            plotly.graph_objects.Figure: Interactive dashboard
        """
        # Create subplots
        methods = list(results_dict.keys())[:6]  # Limit to first 6 methods
        
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=[method.upper() for method in methods],
            specs=[[{"type": "scatter"} for _ in range(3)] for _ in range(2)]
        )
        
        colors = px.colors.qualitative.Set1
        
        for idx, method in enumerate(methods):
            if 'embedding' not in results_dict[method]:
                continue
                
            embedding = results_dict[method]['embedding']
            if embedding.shape[1] < 2:
                continue
                
            row = (idx // 3) + 1
            col = (idx % 3) + 1
            
            if y is not None:
                for class_idx in np.unique(y):
                    mask = y == class_idx
                    fig.add_trace(
                        go.Scatter(
                            x=embedding[mask, 0],
                            y=embedding[mask, 1],
                            mode='markers',
                            name=FASHION_LABELS.get(class_idx, f'Class {class_idx}'),
                            marker=dict(color=colors[class_idx % len(colors)], size=4),
                            showlegend=(idx == 0)  # Only show legend for first subplot
                        ),
                        row=row, col=col
                    )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=embedding[:, 0],
                        y=embedding[:, 1],
                        mode='markers',
                        name=method,
                        marker=dict(size=4),
                        showlegend=False
                    ),
                    row=row, col=col
                )
        
        fig.update_layout(
            title="Dimensionality Reduction Methods Comparison",
            height=800,
            showlegend=True
        )
        
        return fig
    

def create_interactive_plot(X_embedded, y=None, title="Interactive Embedding", 
                          labels=None):
    """
    Convenience function to create interactive 2D plot.
    
    Args:
        X_embedded (np.ndarray): 2D embedded data
        y (np.ndarray, optional): Labels for coloring
        title (str): Plot title
        labels (dict, optional): Label mapping
        
    Returns:
        plotly.graph_objects.Figure: Interactive plot
    """
    visualizer = DimensionalityReductionVisualizer()
    return visualizer.plot_2d_embedding(X_embedded, y, title, labels, interactive=True)


def save_all_plots(results_dict, evaluations, y=None, output_dir='results/figures/'):
    """
    Save all visualization plots to files.
    
    Args:
        results_dict (dict): Dimensionality reduction results
        evaluations (dict): Evaluation results
        y (np.ndarray, optional): Labels
        output_dir (str): Output directory
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    visualizer = DimensionalityReductionVisualizer()
    
    # Multiple embeddings plot
    fig1 = visualizer.plot_multiple_embeddings(results_dict, y, save_path=f'{output_dir}/embeddings_comparison.png')
    
    # Explained variance plot
    fig2 = visualizer.plot_explained_variance(results_dict, save_path=f'{output_dir}/explained_variance.png')
    
    # Performance comparison
    fig3 = visualizer.plot_performance_comparison(evaluations, save_path=f'{output_dir}/performance_comparison.png')
    
    # Runtime comparison
    fig4 = visualizer.plot_runtime_comparison(results_dict, save_path=f'{output_dir}/runtime_comparison.png')
    
    logger.info(f"Saved all plots to {output_dir}")


if __name__ == "__main__":
    # Example usage
    from src.data_loader import load_fashion_mnist
    from src.dimensionality_reducer import DimensionalityReducer
    from src.evaluators import evaluate_all_metrics
    
    # Load data
    X_train, X_test, y_train, y_test = load_fashion_mnist(subset_size=1000)
    
    # Apply dimensionality reduction
    reducer = DimensionalityReducer()
    results = reducer.compare_methods(
        X_train, y_train,
        methods=['pca', 'tsne', 'umap'],
        n_components=2
    )
    
    # Evaluate
    evaluator = evaluate_all_metrics(results, X_train, X_test, y_train, y_test)
    
    # Visualize
    visualizer = DimensionalityReductionVisualizer()
    
    # Multiple embeddings
    fig1 = visualizer.plot_multiple_embeddings(results, y_train)
    plt.show()
    
    # Interactive plot
    if 'umap' in results:
        fig2 = create_interactive_plot(
            results['umap']['embedding'], 
            y_train, 
            "UMAP Interactive Embedding"
        )
        fig2.show()
    
    # Performance comparison
    fig3 = visualizer.plot_performance_comparison(evaluator.metrics)
    plt.show()