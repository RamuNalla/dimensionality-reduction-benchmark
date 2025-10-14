import os
import sys
import argparse
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils import load_results
from visualizers import DimensionalityReductionVisualizer, create_interactive_plot


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate interactive report from experiment results'
    )
    
    parser.add_argument(
        '--results-dir', default='results/',
        help='Directory containing experiment results (default: results/)'
    )
    
    parser.add_argument(
        '--output-file', default='interactive_report.html',
        help='Output HTML file name (default: interactive_report.html)'
    )
    
    parser.add_argument(
        '--include-3d', action='store_true',
        help='Include 3D visualizations'
    )
    
    return parser.parse_args()


def load_experiment_data(results_dir):
    """Load all experiment data."""
    results_file = os.path.join(results_dir, 'results.pkl')
    evaluations_file = os.path.join(results_dir, 'evaluations.pkl')
    
    if not os.path.exists(results_file):
        raise FileNotFoundError(f"Results file not found: {results_file}")
    
    if not os.path.exists(evaluations_file):
        raise FileNotFoundError(f"Evaluations file not found: {evaluations_file}")
    
    results = load_results(results_file)
    evaluations = load_results(evaluations_file)
    
    return results, evaluations


def create_performance_dashboard(evaluations):
    """Create performance comparison dashboard."""
    
    # Prepare data for plotting
    methods = []
    fit_times = []
    silhouette_scores = []
    trustworthiness_scores = []
    continuity_scores = []
    
    for method, eval_dict in evaluations.items():
        if 'error' not in eval_dict:
            methods.append(method.upper())
            fit_times.append(eval_dict.get('fit_time', 0))
            silhouette_scores.append(eval_dict.get('silhouette_score', 0))
            trustworthiness_scores.append(eval_dict.get('trustworthiness', 0))
            continuity_scores.append(eval_dict.get('continuity', 0))
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Runtime Comparison', 'Silhouette Score', 
                       'Trustworthiness', 'Continuity'],
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Runtime comparison (log scale)
    fig.add_trace(
        go.Bar(x=methods, y=fit_times, name='Fit Time',
               marker_color='lightblue', showlegend=False),
        row=1, col=1
    )
    fig.update_yaxes(type="log", row=1, col=1)
    fig.update_yaxes(title_text="Time (seconds, log scale)", row=1, col=1)
    
    # Silhouette score
    fig.add_trace(
        go.Bar(x=methods, y=silhouette_scores, name='Silhouette Score',
               marker_color='lightgreen', showlegend=False),
        row=1, col=2
    )
    fig.update_yaxes(title_text="Silhouette Score", row=1, col=2)
    
    # Trustworthiness
    fig.add_trace(
        go.Bar(x=methods, y=trustworthiness_scores, name='Trustworthiness',
               marker_color='lightcoral', showlegend=False),
        row=2, col=1
    )
    fig.update_yaxes(title_text="Trustworthiness", row=2, col=1)
    
    # Continuity
    fig.add_trace(
        go.Bar(x=methods, y=continuity_scores, name='Continuity',
               marker_color='lightyellow', showlegend=False),
        row=2, col=2
    )
    fig.update_yaxes(title_text="Continuity", row=2, col=2)
    
    fig.update_layout(
        title_text="Performance Metrics Dashboard",
        height=600,
        showlegend=False
    )
    
    return fig

def create_embedding_dashboard(results, sample_labels=None):
    """Create interactive embedding visualization dashboard."""
    
    valid_results = {k: v for k, v in results.items() 
                    if 'embedding' in v and v['embedding'].shape[1] >= 2}
    
    if not valid_results:
        return None
    
    # Limit to first 6 methods for display
    methods = list(valid_results.keys())[:6]
    
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[method.upper() for method in methods],
        specs=[[{"type": "scatter"} for _ in range(3)] for _ in range(2)]
    )
    
    colors = px.colors.qualitative.Set1
    
    for idx, method in enumerate(methods):
        embedding = valid_results[method]['embedding']
        
        row = (idx // 3) + 1
        col = (idx % 3) + 1
        
        if sample_labels is not None:
            # Plot by class
            for class_idx in range(10):  # Fashion-MNIST has 10 classes
                mask = sample_labels == class_idx
                if mask.sum() > 0:  # Only plot if class exists
                    fig.add_trace(
                        go.Scatter(
                            x=embedding[mask, 0],
                            y=embedding[mask, 1],
                            mode='markers',
                            name=f'Class {class_idx}',
                            marker=dict(
                                color=colors[class_idx % len(colors)], 
                                size=4,
                                opacity=0.7
                            ),
                            showlegend=(idx == 0)  # Only show legend for first plot
                        ),
                        row=row, col=col
                    )
        else:
            # Plot without class information
            fig.add_trace(
                go.Scatter(
                    x=embedding[:, 0],
                    y=embedding[:, 1],
                    mode='markers',
                    name=method,
                    marker=dict(size=4, opacity=0.7),
                    showlegend=False
                ),
                row=row, col=col
            )
    
    fig.update_layout(
        title="2D Embeddings Comparison",
        height=800,
        showlegend=True if sample_labels is not None else False
    )
    
    return fig

def create_comparison_table(evaluations):
    """Create interactive comparison table."""
    
    # Prepare data
    table_data = []
    for method, eval_dict in evaluations.items():
        if 'error' not in eval_dict:
            row = {
                'Method': method.upper(),
                'Fit Time (s)': round(eval_dict.get('fit_time', 0), 3),
                'Silhouette Score': round(eval_dict.get('silhouette_score', 0), 3),
                'Trustworthiness': round(eval_dict.get('trustworthiness', 0), 3),
                'Continuity': round(eval_dict.get('continuity', 0), 3),
                'Distance Correlation': round(eval_dict.get('distance_correlation_pearson', 0), 3),
                'Reconstruction Error': round(eval_dict.get('reconstruction_error', 0), 6)
            }
            
            # Add classification accuracy if available
            if 'classification' in eval_dict:
                clf_metrics = eval_dict['classification']
                if 'knn' in clf_metrics and 'error' not in clf_metrics['knn']:
                    row['KNN Accuracy'] = round(clf_metrics['knn'].get('test_accuracy', 0), 3)
            
            table_data.append(row)
    
    df = pd.DataFrame(table_data)
    
    # Create interactive table
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=list(df.columns),
            fill_color='paleturquoise',
            align='left',
            font=dict(size=12, color='black')
        ),
        cells=dict(
            values=[df[col] for col in df.columns],
            fill_color='lavender',
            align='left',
            font=dict(size=11, color='black')
        )
    )])
    
    fig.update_layout(
        title="Performance Metrics Comparison Table",
        height=400
    )
    
    return fig


def create_radar_chart(evaluations):
    """Create radar chart comparing methods across multiple metrics."""
    
    metrics = ['silhouette_score', 'trustworthiness', 'continuity', 'distance_correlation_pearson']
    metric_labels = ['Silhouette Score', 'Trustworthiness', 'Continuity', 'Distance Correlation']
    
    fig = go.Figure()
    
    colors = px.colors.qualitative.Set1
    color_idx = 0
    
    for method, eval_dict in evaluations.items():
        if 'error' not in eval_dict:
            values = []
            for metric in metrics:
                value = eval_dict.get(metric, 0)
                # Normalize to 0-1 scale (assuming all metrics are already 0-1 or -1-1)
                if metric == 'distance_correlation_pearson':
                    value = (value + 1) / 2  # Convert from -1,1 to 0,1
                values.append(max(0, min(1, value)))  # Clamp to 0-1
            
            # Add first value at the end to close the radar chart
            values.append(values[0])
            labels = metric_labels + [metric_labels[0]]
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=labels,
                fill='toself',
                name=method.upper(),
                line_color=colors[color_idx % len(colors)],
                opacity=0.7
            ))
            
            color_idx += 1
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title="Methods Comparison Radar Chart"
    )
    
    return fig


def create_html_report(results, evaluations, output_file, include_3d=False):
    """Create comprehensive HTML report."""
    
    # Create visualizations
    performance_dashboard = create_performance_dashboard(evaluations)
    
    # Try to create embedding dashboard (may not have labels)
    embedding_dashboard = create_embedding_dashboard(results)
    
    comparison_table = create_comparison_table(evaluations)
    radar_chart = create_radar_chart(evaluations)
    
    # Create HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Dimensionality Reduction Comparison Report</title>
        <meta charset="utf-8">
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 40px;
                background-color: #f5f5f5;
            }}
            .header {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 30px;
                border-radius: 10px;
                margin-bottom: 30px;
            }}
            .section {{
                background: white;
                padding: 20px;
                margin-bottom: 20px;
                border-radius: 10px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .chart-container {{
                margin: 20px 0;
            }}
            .summary-box {{
                background: #e3f2fd;
                border-left: 4px solid #2196f3;
                padding: 15px;
                margin: 20px 0;
            }}
            .metric-highlight {{
                display: inline-block;
                background: #4caf50;
                color: white;
                padding: 5px 10px;
                border-radius: 5px;
                margin: 5px;
            }}
        </style>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    </head>
    <body>
        <div class="header">
            <h1>Dimensionality Reduction Comparison Report</h1>
            <p>Comprehensive analysis of {len([k for k, v in evaluations.items() if 'error' not in v])} dimensionality reduction methods on Fashion-MNIST dataset</p>
        </div>
        
        <div class="section">
            <h2>Executive Summary</h2>
            <div class="summary-box">
                <p><strong>Dataset:</strong> Fashion-MNIST with {list(results.values())[0]['embedding'].shape[0] if results else 'N/A'} samples</p>
                <p><strong>Methods Compared:</strong> {', '.join([k.upper() for k, v in evaluations.items() if 'error' not in v])}</p>
                <p><strong>Evaluation Metrics:</strong> Runtime, Silhouette Score, Trustworthiness, Continuity, Distance Correlation</p>
            </div>
        </div>
        
        <div class="section">
            <h2>Performance Dashboard</h2>
            <div id="performance-dashboard" class="chart-container"></div>
        </div>
        
        <div class="section">
            <h2>Methods Comparison</h2>
            <div id="radar-chart" class="chart-container"></div>
        </div>
        
        <div class="section">
            <h2>Detailed Metrics Table</h2>
            <div id="comparison-table" class="chart-container"></div>
        </div>
    """
    
    if embedding_dashboard:
        html_content += """
        <div class="section">
            <h2>2D Embeddings Visualization</h2>
            <div id="embedding-dashboard" class="chart-container"></div>
        </div>
        """
    
    # Add recommendations section
    html_content += create_recommendations_section(evaluations)
    
    html_content += """
        <div class="section">
            <h2>Method Details</h2>
    """
    
    # Add method descriptions
    method_descriptions = {
        'pca': 'Principal Component Analysis - Linear method that finds directions of maximum variance',
        'lda': 'Linear Discriminant Analysis - Supervised method that maximizes class separation',
        'ica': 'Independent Component Analysis - Finds statistically independent components',
        'tsne': 't-SNE - Non-linear method excellent for visualization, preserves local structure',
        'umap': 'UMAP - Fast non-linear method that preserves both local and global structure',
        'kernel_pca': 'Kernel PCA - Non-linear extension of PCA using kernel tricks',
        'isomap': 'Isomap - Preserves geodesic distances on data manifold',
        'lle': 'Locally Linear Embedding - Preserves local neighborhood relationships',
        'autoencoder': 'Autoencoder - Neural network approach for non-linear dimensionality reduction'
    }
    
    for method, eval_dict in evaluations.items():
        if 'error' not in eval_dict:
            description = method_descriptions.get(method, 'Advanced dimensionality reduction method')
            fit_time = eval_dict.get('fit_time', 0)
            silhouette = eval_dict.get('silhouette_score', 0)
            
            html_content += f"""
            <div style="border: 1px solid #ddd; padding: 15px; margin: 10px 0; border-radius: 5px;">
                <h3>{method.upper()}</h3>
                <p>{description}</p>
                <div style="display: flex; gap: 15px;">
                    <span class="metric-highlight">Runtime: {fit_time:.2f}s</span>
                    <span class="metric-highlight">Silhouette: {silhouette:.3f}</span>
                </div>
            </div>
            """
    
    html_content += """
        </div>
        
        <div class="section">
            <h2>Usage Guidelines</h2>
            <ul>
                <li><strong>For Visualization:</strong> Use t-SNE or UMAP for best cluster separation</li>
                <li><strong>For Speed:</strong> Use PCA or SVD for fastest processing</li>
                <li><strong>For Machine Learning:</strong> Use LDA (supervised) or PCA for downstream tasks</li>
                <li><strong>For Non-linear Data:</strong> Use UMAP, t-SNE, or Kernel PCA</li>
                <li><strong>For Interpretability:</strong> Use PCA for easily interpretable components</li>
            </ul>
        </div>
        
        <script>
    """
    
    # Add JavaScript for plotting
    html_content += f"""
            Plotly.newPlot('performance-dashboard', {performance_dashboard.to_json()});
            Plotly.newPlot('radar-chart', {radar_chart.to_json()});
            Plotly.newPlot('comparison-table', {comparison_table.to_json()});
    """
    
    if embedding_dashboard:
        html_content += f"""
            Plotly.newPlot('embedding-dashboard', {embedding_dashboard.to_json()});
        """
    
    html_content += """
        </script>
    </body>
    </html>
    """
    
    # Save HTML file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Interactive report saved to: {output_file}")


def create_recommendations_section(evaluations):
    """Create recommendations based on evaluation results."""
    
    # Find best methods for different criteria
    valid_evals = {k: v for k, v in evaluations.items() if 'error' not in v}
    
    if not valid_evals:
        return "<div class='section'><h2>Recommendations</h2><p>No valid results to analyze.</p></div>"
    
    # Best speed
    fastest = min(valid_evals.items(), key=lambda x: x[1].get('fit_time', float('inf')))
    
    # Best clustering
    best_clustering = max(valid_evals.items(), key=lambda x: x[1].get('silhouette_score', -1))
    
    # Best preservation
    best_preservation = max(valid_evals.items(), key=lambda x: x[1].get('trustworthiness', -1))
    
    # Most balanced (combining multiple metrics)
    def balanced_score(eval_dict):
        silhouette = eval_dict.get('silhouette_score', 0)
        trustworthiness = eval_dict.get('trustworthiness', 0)
        continuity = eval_dict.get('continuity', 0)
        # Normalize fit time (lower is better)
        fit_time = eval_dict.get('fit_time', 1)
        time_score = 1 / (1 + fit_time)  # Convert to 0-1 where higher is better
        
        return (silhouette + trustworthiness + continuity + time_score) / 4
    
    most_balanced = max(valid_evals.items(), key=lambda x: balanced_score(x[1]))
    
    recommendations_html = f"""
    <div class="section">
        <h2>Recommendations</h2>
        <div class="summary-box">
            <h3>🏆 Top Performers by Category</h3>
            <p><strong>Fastest Method:</strong> {fastest[0].upper()} ({fastest[1].get('fit_time', 0):.2f}s)</p>
            <p><strong>Best Clustering:</strong> {best_clustering[0].upper()} (Silhouette: {best_clustering[1].get('silhouette_score', 0):.3f})</p>
            <p><strong>Best Structure Preservation:</strong> {best_preservation[0].upper()} (Trustworthiness: {best_preservation[1].get('trustworthiness', 0):.3f})</p>
            <p><strong>Most Balanced:</strong> {most_balanced[0].upper()} (Overall score: {balanced_score(most_balanced[1]):.3f})</p>
        </div>
    </div>
    """
    
    return recommendations_html


def main():
    """Main entry point."""
    args = parse_arguments()
    
    try:
        print("Loading experiment results...")
        results, evaluations = load_experiment_data(args.results_dir)
        
        print(f"Found results for {len(results)} methods")
        print(f"Valid evaluations for {len([k for k, v in evaluations.items() if 'error' not in v])} methods")
        
        print("Generating interactive report...")
        output_path = os.path.join(args.results_dir, args.output_file)
        create_html_report(results, evaluations, output_path, args.include_3d)
        
        print("\n" + "="*60)
        print("INTERACTIVE REPORT GENERATED!")
        print("="*60)
        print(f"Report saved to: {output_path}")
        print(f"Open the file in your web browser to view the interactive dashboard.")
        
        # Print quick summary
        valid_methods = [k for k, v in evaluations.items() if 'error' not in v]
        if valid_methods:
            print(f"\nMethods analyzed: {', '.join([m.upper() for m in valid_methods])}")
            
            # Find best overall method
            best_method = max(
                [(k, v) for k, v in evaluations.items() if 'error' not in v],
                key=lambda x: x[1].get('silhouette_score', 0)
            )
            print(f"Best performing method (by silhouette score): {best_method[0].upper()}")
        
        return 0
        
    except Exception as e:
        print(f"Error generating report: {str(e)}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    
