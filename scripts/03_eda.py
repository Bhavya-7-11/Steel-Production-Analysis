"""
Script 03: Exploratory Data Analysis (EDA)
Generate comprehensive visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Optional

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

def plot_correlation_matrix(df: pd.DataFrame, save_path: Optional[str] = None):
    """Create and save correlation heatmap."""
    numeric_df = df.select_dtypes(include=[np.number])
    correlation = numeric_df.corr()
    
    plt.figure(figsize=(14, 12))
    sns.heatmap(correlation, annot=False, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()

def plot_feature_distributions(df: pd.DataFrame, save_path: Optional[str] = None):
    """Plot histograms for all numerical features."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    n_cols = min(len(numeric_cols), 20)  # Limit to 20 features for readability
    
    n_rows = (n_cols + 3) // 4
    fig, axes = plt.subplots(n_rows, 4, figsize=(16, n_rows * 3))
    axes = axes.flatten() if n_cols > 1 else [axes]
    
    for idx, col in enumerate(numeric_cols[:n_cols]):
        ax = axes[idx]
        ax.hist(df[col].dropna(), bins=50, edgecolor='black', alpha=0.7)
        ax.set_title(f'{col}', fontsize=10, fontweight='bold')
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_cols, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Feature Distributions', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()

def plot_boxplots(df: pd.DataFrame, save_path: Optional[str] = None):
    """Create box plots for outlier detection."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    n_cols = min(len(numeric_cols), 20)
    
    n_rows = (n_cols + 3) // 4
    fig, axes = plt.subplots(n_rows, 4, figsize=(16, n_rows * 3))
    axes = axes.flatten() if n_cols > 1 else [axes]
    
    for idx, col in enumerate(numeric_cols[:n_cols]):
        ax = axes[idx]
        ax.boxplot(df[col].dropna(), vert=True)
        ax.set_title(f'{col}', fontsize=10, fontweight='bold')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_cols, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Box Plots for Outlier Detection', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()

def plot_target_distribution(df: pd.DataFrame, target_col: str = 'output', 
                            save_path: Optional[str] = None):
    """Visualize target variable distribution."""
    if target_col not in df.columns:
        print(f"Warning: Target column '{target_col}' not found")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    axes[0].hist(df[target_col].dropna(), bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0].set_title(f'Distribution of {target_col}', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Value')
    axes[0].set_ylabel('Frequency')
    axes[0].grid(True, alpha=0.3)
    
    # Box plot
    axes[1].boxplot(df[target_col].dropna(), vert=True)
    axes[1].set_title(f'Box Plot of {target_col}', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Value')
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle('Target Variable Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()

def plot_pairplot(df: pd.DataFrame, target_col: str = 'output', n_features: int = 5,
                 save_path: Optional[str] = None):
    """Create pair plot for feature relationships."""
    # Select top correlated features with target
    if target_col in df.columns:
        correlations = df.corr()[target_col].abs().sort_values(ascending=False)
        top_features = correlations.head(n_features + 1).index.tolist()  # +1 for target itself
    else:
        top_features = df.select_dtypes(include=[np.number]).columns[:n_features].tolist()
    
    pairplot = sns.pairplot(df[top_features], diag_kind='hist', corner=True)
    pairplot.fig.suptitle('Feature Pair Plot (Top Correlated)', y=1.02, fontsize=16, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()

def generate_eda_report(df: pd.DataFrame) -> dict:
    """Generate comprehensive EDA report."""
    report = {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes.to_dict(),
        'missing': df.isnull().sum().to_dict(),
        'statistics': df.describe().to_dict()
    }
    
    print("\n=== EDA Report ===")
    print(f"Dataset Shape: {report['shape']}")
    print(f"Number of Features: {len(report['columns'])}")
    print(f"Missing Values: {sum(report['missing'].values())}")
    
    return report

def main():
    """Main execution function."""
    # Define paths (updated to use processed data from script 02)
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_path = os.path.join(base_path, 'data', 'processed_train_data.csv')
    figures_path = os.path.join(base_path, 'figures')
    os.makedirs(figures_path, exist_ok=True)
    
    # Load data
    print("Loading training data...")
    train_df = pd.read_csv(train_path)
    print(f"Data shape: {train_df.shape}")
    
    # Generate EDA report
    report = generate_eda_report(train_df)
    
    # Create visualizations
    print("\nGenerating visualizations...")
    
    print("1. Correlation matrix...")
    plot_correlation_matrix(train_df, 
                           os.path.join(figures_path, '01_correlation_matrix.png'))
    
    print("2. Feature distributions...")
    plot_feature_distributions(train_df,
                              os.path.join(figures_path, '02_feature_distributions.png'))
    
    print("3. Box plots...")
    plot_boxplots(train_df,
                 os.path.join(figures_path, '03_boxplots.png'))
    
    print("4. Target distribution...")
    plot_target_distribution(train_df,
                            save_path=os.path.join(figures_path, '04_target_distribution.png'))
    
    print("5. Pair plot...")
    plot_pairplot(train_df,
                 save_path=os.path.join(figures_path, '05_pairplot.png'))
    
    print("\n✓ EDA completed successfully!")
    print(f"All figures saved to: {figures_path}")

if __name__ == "__main__":
    main()