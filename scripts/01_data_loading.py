"""
Script 01: Data Loading Module
Steel Production Analysis Project
"""

import pandas as pd
import numpy as np
import os
from typing import Tuple

def load_steel_data(train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load steel production training and test datasets.
    
    Args:
        train_path: Path to training data CSV
        test_path: Path to test data CSV
        
    Returns:
        Tuple of (train_df, test_df)
    """
    try:
        print("Loading training data...")
        train_df = pd.read_csv(train_path)
        print(f"Training data shape: {train_df.shape}")
        
        print("Loading test data...")
        test_df = pd.read_csv(test_path)
        print(f"Test data shape: {test_df.shape}")
        
        # Display basic info
        print("\n=== Training Data Info ===")
        print(train_df.info())
        print("\n=== First Few Rows ===")
        print(train_df.head())
        
        # Check for target column
        if 'output' in train_df.columns:
            print(f"\nTarget column 'output' found")
            print(f"Target range: [{train_df['output'].min():.2f}, {train_df['output'].max():.2f}]")
        else:
            print("\nWarning: 'output' column not found. Available columns:")
            print(train_df.columns.tolist())
            
        return train_df, test_df
        
    except FileNotFoundError as e:
        print(f"Error: Data file not found - {e}")
        raise
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

def get_data_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate comprehensive statistics for the dataset.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with statistics
    """
    stats = pd.DataFrame({
        'count': df.count(),
        'mean': df.mean(numeric_only=True),
        'std': df.std(numeric_only=True),
        'min': df.min(numeric_only=True),
        '25%': df.quantile(0.25, numeric_only=True),
        '50%': df.quantile(0.50, numeric_only=True),
        '75%': df.quantile(0.75, numeric_only=True),
        'max': df.max(numeric_only=True),
        'missing': df.isnull().sum(),
        'missing_pct': (df.isnull().sum() / len(df)) * 100
    })
    
    return stats

def main():
    """Main execution function."""
    # Define paths
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_path = os.path.join(base_path, 'data', 'normalized_train_data.csv')
    test_path = os.path.join(base_path, 'data', 'normalized_test_data.csv')
    
    # Load data
    train_df, test_df = load_steel_data(train_path, test_path)
    
    # Generate statistics
    print("\n=== Training Data Statistics ===")
    train_stats = get_data_statistics(train_df)
    print(train_stats)
    
    # Save statistics
    results_path = os.path.join(base_path, 'results')
    os.makedirs(results_path, exist_ok=True)
    train_stats.to_csv(os.path.join(results_path, 'data_statistics.csv'))
    print(f"\nStatistics saved to: {os.path.join(results_path, 'data_statistics.csv')}")
    
    print("\n✓ Data loading completed successfully!")

if __name__ == "__main__":
    main()