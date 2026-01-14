"""
Script 05: Results Analysis and Visualization
Generate comprehensive comparison plots and analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import List

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

def load_results(results_path: str, predictions_path: str) -> tuple:
    """Load performance metrics and predictions."""
    metrics_df = pd.read_csv(results_path)
    
    predictions = {}
    for file in os.listdir(predictions_path):
        if file.endswith('.csv'):
            model_name = file.replace('_predictions.csv', '').replace('_', ' ').title()
            pred_df = pd.read_csv(os.path.join(predictions_path, file))
            predictions[model_name] = pred_df
    
    return metrics_df, predictions

def plot_model_comparison(metrics_df: pd.DataFrame, save_path: str):
    """Create bar plots with error bars for model comparison."""
    metrics = ['rmse', 'mae', 'r2']
    titles = ['Root Mean Square Error (RMSE)', 'Mean Absolute Error (MAE)', 'R² Score']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx]
        
        # Create bar plot
        bars = ax.bar(metrics_df['model'], metrics_df[metric], 
                     color='steelblue', alpha=0.7, edgecolor='black')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}',
                   ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('Model', fontsize=11, fontweight='bold')
        ax.set_ylabel(title, fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

def plot_predictions_vs_actual(predictions: dict, save_path: str):
    """Create scatter plots of predictions vs actual values."""
    n_models = len(predictions)
    n_cols = 3
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
    axes = axes.flatten() if n_models > 1 else [axes]
    
    for idx, (model_name, pred_df) in enumerate(predictions.items()):
        ax = axes[idx]
        
        # Scatter plot
        ax.scatter(pred_df['y_true'], pred_df['y_pred'], 
                  alpha=0.5, s=20, edgecolors='black', linewidth=0.5)
        
        # Perfect prediction line
        min_val = min(pred_df['y_true'].min(), pred_df['y_pred'].min())
        max_val = max(pred_df['y_true'].max(), pred_df['y_pred'].max())
        ax.plot([min_val, max_val], [min_val, max_val], 
               'r--', linewidth=2, label='Perfect Prediction')
        
        # Calculate R²
        r2 = np.corrcoef(pred_df['y_true'], pred_df['y_pred'])[0, 1] ** 2
        
        ax.set_xlabel('Actual Values', fontsize=10, fontweight='bold')
        ax.set_ylabel('Predicted Values', fontsize=10, fontweight='bold')
        ax.set_title(f'{model_name}\n(R² = {r2:.4f})', fontsize=11, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_models, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Predictions vs Actual Values', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

def plot_residuals(predictions: dict, save_path: str):
    """Create residual plots for each model."""
    n_models = len(predictions)
    n_cols = 3
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
    axes = axes.flatten() if n_models > 1 else [axes]
    
    for idx, (model_name, pred_df) in enumerate(predictions.items()):
        ax = axes[idx]
        
        # Calculate residuals
        residuals = pred_df['y_true'] - pred_df['y_pred']
        
        # Residual plot
        ax.scatter(pred_df['y_pred'], residuals, 
                  alpha=0.5, s=20, edgecolors='black', linewidth=0.5)
        
        # Zero line
        ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
        
        # Calculate statistics
        mean_res = residuals.mean()
        std_res = residuals.std()
        
        ax.set_xlabel('Predicted Values', fontsize=10, fontweight='bold')
        ax.set_ylabel('Residuals', fontsize=10, fontweight='bold')
        ax.set_title(f'{model_name}\n(Mean: {mean_res:.4f}, Std: {std_res:.4f})', 
                    fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_models, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Residual Analysis', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

def plot_error_distribution(predictions: dict, save_path: str):
    """Plot error distribution for each model."""
    n_models = len(predictions)
    n_cols = 3
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
    axes = axes.flatten() if n_models > 1 else [axes]
    
    for idx, (model_name, pred_df) in enumerate(predictions.items()):
        ax = axes[idx]
        
        # Calculate errors
        errors = pred_df['y_true'] - pred_df['y_pred']
        
        # Histogram
        ax.hist(errors, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
        
        # Add vertical line at zero
        ax.axvline(x=0, color='r', linestyle='--', linewidth=2)
        
        ax.set_xlabel('Prediction Error', fontsize=10, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=10, fontweight='bold')
        ax.set_title(f'{model_name}', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    
    # Hide unused subplots
    for idx in range(n_models, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Error Distribution', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

def create_performance_table(metrics_df: pd.DataFrame, save_path: str):
    """Create formatted performance comparison table."""
    # Create figure for table
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('tight')
    ax.axis('off')
    
    # Format the dataframe
    table_data = metrics_df.copy()
    table_data['rmse'] = table_data['rmse'].apply(lambda x: f'{x:.4f}')
    table_data['mae'] = table_data['mae'].apply(lambda x: f'{x:.4f}')
    table_data['r2'] = table_data['r2'].apply(lambda x: f'{x:.4f}')
    table_data['inference_time'] = table_data['inference_time'].apply(lambda x: f'{x:.4f}s')
    
    # Create table
    table = ax.table(cellText=table_data.values,
                    colLabels=table_data.columns,
                    cellLoc='center',
                    loc='center',
                    colColours=['lightgray'] * len(table_data.columns))
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header
    for i in range(len(table_data.columns)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.title('Model Performance Comparison', fontsize=14, fontweight='bold', pad=20)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

def generate_summary_report(metrics_df: pd.DataFrame) -> str:
    """Generate text summary of results."""
    best_rmse = metrics_df.loc[metrics_df['rmse'].idxmin()]
    best_mae = metrics_df.loc[metrics_df['mae'].idxmin()]
    best_r2 = metrics_df.loc[metrics_df['r2'].idxmax()]
    fastest = metrics_df.loc[metrics_df['inference_time'].idxmin()]
    
    report = f"""
=== ANALYSIS SUMMARY ===

Best RMSE: {best_rmse['model']} ({best_rmse['rmse']:.4f})
Best MAE: {best_mae['model']} ({best_mae['mae']:.4f})
Best R²: {best_r2['model']} ({best_r2['r2']:.4f})
Fastest Inference: {fastest['model']} ({fastest['inference_time']:.4f}s)

=== ALL MODELS ===
{metrics_df.to_string(index=False)}
"""
    
    return report

def main():
    """Main execution function."""
    # Define paths
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_path = os.path.join(base_path, 'results', 'performance_metrics.csv')
    predictions_path = os.path.join(base_path, 'results', 'model_predictions')
    figures_path = os.path.join(base_path, 'figures')
    
    # Load results
    print("Loading results...")
    metrics_df, predictions = load_results(results_path, predictions_path)
    
    print(f"Loaded {len(predictions)} model predictions")
    
    # Generate visualizations
    print("\nGenerating analysis visualizations...")
    
    print("1. Model comparison...")
    plot_model_comparison(metrics_df, 
                         os.path.join(figures_path, '06_model_comparison.png'))
    
    print("2. Predictions vs actual...")
    plot_predictions_vs_actual(predictions,
                              os.path.join(figures_path, '07_predictions_vs_actual.png'))
    
    print("3. Residual analysis...")
    plot_residuals(predictions,
                  os.path.join(figures_path, '08_residual_analysis.png'))
    
    print("4. Error distribution...")
    plot_error_distribution(predictions,
                           os.path.join(figures_path, '09_error_distribution.png'))
    
    print("5. Performance table...")
    create_performance_table(metrics_df,
                            os.path.join(figures_path, '10_performance_table.png'))
    
    # Generate summary report
    summary = generate_summary_report(metrics_df)
    print(summary)
    
    # Save summary to file
    summary_path = os.path.join(base_path, 'results', 'analysis_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(summary)
    print(f"\nSummary saved to: {summary_path}")
    
    print("\n✓ Results analysis completed successfully!")

if __name__ == "__main__":
    main()