# Dimensionality Reduction Benchmark 

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A comprehensive comparison of 10+ dimensionality reduction techniques on the Fashion-MNIST dataset. This project provides quantitative evaluation metrics and performance analysis to help data scientists choose the best dimensionality reduction method for their use case.

## Project Overview

This benchmark evaluates dimensionality reduction methods across multiple criteria:
- **Performance**: Clustering quality, classification accuracy
- **Efficiency**: Runtime, memory usage, scalability
- **Visualization**: 2D/3D plots, interactive dashboards

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/RamuNalla/dimensionality-reduction-benchmark.git
cd dimensionality-reduction-benchmark

# Create virtual environment 
python -m venv venv
source venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

```

### Run Complete Analysis

```bash
# Run all experiments with default settings (5000 samples)
python scripts/run_all_experiments.py

# Generate interactive HTML report
python scripts/generate_report.py

```

## Algorithms Compared

### Linear Methods
| Method | Description | Key Strengths |
|--------|-------------|---------------|
| **PCA** | Principal Component Analysis | Fast, interpretable, explained variance |
| **LDA** | Linear Discriminant Analysis | Supervised, maximizes class separation |
| **ICA** | Independent Component Analysis | Finds independent sources |
| **SVD** | Truncated Singular Value Decomposition | Memory efficient, sparse data |

### Non-linear Methods
| Method | Description | Key Strengths |
|--------|-------------|---------------|
| **t-SNE** | t-Distributed Stochastic Neighbor Embedding | Excellent visualization, local structure |
| **UMAP** | Uniform Manifold Approximation and Projection | Fast, global + local structure |
| **Kernel PCA** | Non-linear PCA with kernel trick | Non-linear, invertible |
| **Isomap** | Isometric Mapping | Preserves geodesic distances |
| **LLE** | Locally Linear Embedding | Local neighborhood preservation |

### Neural Network-based
| Method | Description | Key Strengths |
|--------|-------------|---------------|
| **Autoencoder** | Deep neural network encoder-decoder | Highly flexible, customizable architecture |

## Evaluation Metrics

### Quality Metrics
- **Silhouette Score**: Measures clustering quality (-1 to 1, higher better)
- **Trustworthiness**: Local neighborhood preservation (0 to 1, higher better)
- **Continuity**: Smoothness of the mapping (0 to 1, higher better)
- **Distance Correlation**: Pearson/Spearman correlation of pairwise distances

### Performance Metrics
- **Runtime**: Training and inference time
- **Memory Usage**: Peak memory consumption
- **Classification Accuracy**: k-NN and Logistic Regression on reduced data
- **Reconstruction Error**: MSE between original and reconstructed data

### Statistical Metrics
- **Explained Variance Ratio**: Proportion of variance retained (linear methods)
- **Stress**: Goodness of distance preservation (MDS-like methods)

### Key Findings
- **Best for visualization**: t-SNE and UMAP show superior cluster separation
- **Best for speed**: PCA and SVD are fastest for large datasets (>10k samples)
- **Best for downstream ML**: LDA maintains highest classification accuracy
- **Most versatile**: UMAP balances speed, accuracy, and visualization quality

## Project Structure

```
dimensionality-reduction-benchmark/
├── 📄 README.md                    # This file
├── ⚙️  requirements.txt             # Dependencies
├── 📦 setup.py                     # Package installation
├── 🗂️  src/                        # Core implementation
│   ├── 📊 data_loader.py           # Fashion-MNIST data handling  
│   ├── 🔧 dimensionality_reducers.py # All DR algorithms
│   ├── 📏 evaluators.py            # Evaluation metrics
│   ├── 📈 visualizers.py           # Plotting and visualization
│   └── 🛠️  utils.py                # Utility functions
├── 📓 notebooks/                   # Jupyter analysis notebooks
│   ├── 01_data_exploration.ipynb  # Dataset analysis
│   ├── 02_linear_methods.ipynb    # PCA, LDA, ICA, SVD
│   ├── 03_nonlinear_methods.ipynb # t-SNE, UMAP, Kernel PCA
│   ├── 04_neural_methods.ipynb    # Autoencoder experiments
│   └── 05_comparison_analysis.ipynb # Comprehensive comparison
├── 🤖 scripts/                     # Automation scripts
│   ├── run_all_experiments.py     # Main experiment runner
│   └── generate_report.py         # Interactive report generator
├── 📊 results/                     # Generated outputs
│   ├── figures/                   # Plots and visualizations
│   ├── metrics/                   # Performance data
│   └── models/                    # Saved models
└── 🧪 tests/                       # Unit tests
    ├── test_data_loader.py
    ├── test_reducers.py
    └── test_evaluators.py
```

## Advanced Usage

### Custom Method Parameters

```python
# Run with custom hyperparameters
python scripts/run_all_experiments.py \
    --methods tsne umap \
    --subset-size 5000 \
    --output-dir custom_results/

# Or modify parameters in code
from src.dimensionality_reducers import DimensionalityReducer

reducer = DimensionalityReducer()
results = reducer.compare_methods(
    X_train, y_train,
    methods=['tsne', 'umap'],
    n_components=2,
    tsne={'perplexity': 50, 'n_iter': 2000},
    umap={'n_neighbors': 30, 'min_dist': 0.01}
)
```

### Memory and Performance Optimization

```python
# Check system requirements before running
from src.utils import check_system_requirements

requirements = check_system_requirements(
    n_samples=10000, 
    n_features=784, 
    methods=['pca', 'tsne', 'umap']
)

print(f"Feasible: {requirements['feasible']}")
print(f"Recommendation: {requirements['recommendation']}")
```

### Custom Dataset

```python
from src.dimensionality_reducers import DimensionalityReducer
from src.evaluators import evaluate_all_metrics

# Load your custom dataset
X_train, X_test, y_train, y_test = load_your_data()

# Run comparison
reducer = DimensionalityReducer()
results = reducer.compare_methods(X_train, y_train)

# Evaluate
evaluator = evaluate_all_metrics(results, X_train, X_test, y_train, y_test)
```

## Interactive Visualizations

The project generates several types of visualizations:

### Static Plots (Matplotlib/Seaborn)
- Multi-panel embedding comparisons
- Performance bar charts
- Runtime analysis
- Explained variance plots

### Interactive Dashboards (Plotly)
- Zoomable scatter plots with class highlighting
- Hover tooltips with sample information
- Interactive comparison tables
- 3D visualizations

### Sample Visualization Code
```python
from src.visualizers import create_interactive_plot

# Create interactive 2D embedding plot
fig = create_interactive_plot(
    embedding_2d, 
    labels, 
    title="UMAP Projection of Fashion-MNIST"
)
fig.show()
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

