"""
Script 02: Enhanced Data Preprocessing
Optimized for maximum model performance
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

def smart_outlier_treatment(df: pd.DataFrame, method: str = 'cap', multiplier: float = 2.5) -> pd.DataFrame:
    """
    Smart outlier treatment - uses higher multiplier to preserve more data
    """
    df_treated = df.copy()
    numeric_cols = df_treated.select_dtypes(include=[np.number]).columns
    
    outliers_info = []
    for col in numeric_cols:
        Q1 = df_treated[col].quantile(0.25)
        Q3 = df_treated[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        n_outliers_before = ((df_treated[col] < lower_bound) | (df_treated[col] > upper_bound)).sum()
        
        if method == 'cap':
            df_treated[col] = df_treated[col].clip(lower=lower_bound, upper=upper_bound)
        
        outliers_info.append({
            'column': col,
            'n_outliers': n_outliers_before,
            'pct': (n_outliers_before / len(df)) * 100
        })
    
    print(f"\nOutliers treated using '{method}' method (IQR multiplier: {multiplier})")
    print(pd.DataFrame(outliers_info).to_string())
    
    return df_treated

def remove_low_variance_features(df: pd.DataFrame, threshold: float = 0.01) -> pd.DataFrame:
    """Remove features with very low variance"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    variances = df[numeric_cols].var()
    low_var_cols = variances[variances < threshold].index.tolist()
    
    if low_var_cols:
        print(f"\nRemoving {len(low_var_cols)} low-variance features: {low_var_cols}")
        df = df.drop(columns=low_var_cols)
    else:
        print("\nNo low-variance features found")
    
    return df

def handle_missing_values_advanced(df: pd.DataFrame) -> pd.DataFrame:
    """Advanced missing value handling"""
    missing_before = df.isnull().sum().sum()
    
    if missing_before == 0:
        print("No missing values found")
        return df
    
    df_clean = df.copy()
    
    for col in df_clean.columns:
        if df_clean[col].isnull().any():
            # Use median for robustness
            df_clean[col].fillna(df_clean[col].median(), inplace=True)
    
    missing_after = df_clean.isnull().sum().sum()
    print(f"Missing values: {missing_before} → {missing_after}")
    
    return df_clean

def validate_data(df: pd.DataFrame, name: str = "Dataset"):
    """Comprehensive data validation"""
    print(f"\n=== {name} Validation ===")
    print(f"Shape: {df.shape}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    print(f"Duplicate rows: {df.duplicated().sum()}")
    print(f"Infinite values: {np.isinf(df.select_dtypes(include=[np.number])).sum().sum()}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

def main():
    """Main preprocessing pipeline"""
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_path = os.path.join(base_path, 'data', 'normalized_train_data.csv')
    test_path = os.path.join(base_path, 'data', 'normalized_test_data.csv')
    
    print("="*70)
    print("ENHANCED DATA PREPROCESSING")
    print("="*70)
    
    # Load data
    print("\nLoading data...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    print(f"Original - Train: {train_df.shape}, Test: {test_df.shape}")
    
    # Combine for consistent preprocessing
    df = pd.concat([train_df, test_df], ignore_index=True)
    print(f"Combined shape: {df.shape}")
    
    # 1. Remove duplicates
    print("\n--- Step 1: Remove Duplicates ---")
    initial_rows = len(df)
    df = df.drop_duplicates()
    removed = initial_rows - len(df)
    print(f"Removed {removed} duplicates ({removed/initial_rows*100:.2f}%)")
    
    # 2. Handle missing values
    print("\n--- Step 2: Handle Missing Values ---")
    df = handle_missing_values_advanced(df)
    
    # 3. Smart outlier treatment (use capping, not removal)
    print("\n--- Step 3: Outlier Treatment ---")
    df = smart_outlier_treatment(df, method='cap', multiplier=2.5)
    
    # 4. Remove low variance features
    print("\n--- Step 4: Remove Low Variance Features ---")
    df = remove_low_variance_features(df, threshold=0.001)
    
    # 5. Validate final data
    validate_data(df, "Processed Dataset")
    
    # 6. Split data (80-20)
    print("\n--- Step 5: Train-Test Split ---")
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)
    print(f"Train: {train_df.shape}, Test: {test_df.shape}")
    
    # Save processed data
    processed_train_path = os.path.join(base_path, 'data', 'processed_train_data.csv')
    processed_test_path = os.path.join(base_path, 'data', 'processed_test_data.csv')
    
    train_df.to_csv(processed_train_path, index=False)
    test_df.to_csv(processed_test_path, index=False)
    
    print(f"\nSaved processed data:")
    print(f"  Train: {processed_train_path}")
    print(f"  Test: {processed_test_path}")
    
    # Generate preprocessing report
    report = f"""
PREPROCESSING SUMMARY
=====================
Original combined size: {initial_rows} rows
Final size: {len(df)} rows
Features: {df.shape[1]}
Train set: {len(train_df)} rows
Test set: {len(test_df)} rows

Data Quality:
- Missing values: {df.isnull().sum().sum()}
- Duplicates: {df.duplicated().sum()}
- Infinite values: {np.isinf(df.select_dtypes(include=[np.number])).sum().sum()}
"""
    
    print(report)
    
    # Save report
    report_path = os.path.join(base_path, 'results', 'preprocessing_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    
    print("\n✓ Enhanced preprocessing completed successfully!")

if __name__ == "__main__":
    main()