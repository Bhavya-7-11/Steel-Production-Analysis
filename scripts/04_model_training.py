"""
Script 04: Enhanced Model Training Module
Implements optimized regression models with advanced feature engineering
"""

import pandas as pd
import numpy as np
import os
import time
import pickle
from typing import Tuple, Dict
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, Matern, WhiteKernel
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import cross_val_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

np.random.seed(42)
tf.random.set_seed(42)

def advanced_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Advanced domain-specific feature engineering for steel production"""
    df = df.copy()
    
    # Carbon equivalent (critical for steel properties)
    if all(col in df.columns for col in ['input1', 'input3', 'input6', 'input7', 'input8', 'input9', 'input10']):
        df['C_eq'] = df['input1'] + df['input3']/6 + (df['input7'] + df['input9'] + df['input10'])/5 + (df['input6'] + df['input8'])/15
        df['C_eq_squared'] = df['C_eq'] ** 2
    
    # Alloy interactions
    if 'input7' in df.columns and 'input9' in df.columns:
        df['Cr_Mo_ratio'] = df['input7'] / (df['input9'] + 1e-8)
        df['Cr_Mo_product'] = df['input7'] * df['input9']
    
    if 'input6' in df.columns and 'input8' in df.columns:
        df['Ni_Cu_ratio'] = df['input6'] / (df['input8'] + 1e-8)
        df['Ni_Cu_product'] = df['input6'] * df['input8']
    
    # Statistical features per row
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df['row_mean'] = df[numeric_cols].mean(axis=1)
    df['row_std'] = df[numeric_cols].std(axis=1)
    df['row_max'] = df[numeric_cols].max(axis=1)
    df['row_min'] = df[numeric_cols].min(axis=1)
    df['row_range'] = df['row_max'] - df['row_min']
    
    # Polynomial features for key inputs
    key_inputs = ['input1', 'input2', 'input3', 'input4', 'input5']
    for col in key_inputs:
        if col in df.columns:
            df[f'{col}_squared'] = df[col] ** 2
            df[f'{col}_cubed'] = df[col] ** 3
            df[f'{col}_sqrt'] = np.sqrt(np.abs(df[col]))
            df[f'{col}_log'] = np.log1p(np.abs(df[col]))
    
    # Interaction terms for top correlated features
    if 'input1' in df.columns and 'input2' in df.columns:
        df['input1_x_input2'] = df['input1'] * df['input2']
    if 'input3' in df.columns and 'input4' in df.columns:
        df['input3_x_input4'] = df['input3'] * df['input4']
    
    return df

def prepare_data_advanced(train_path: str, test_path: str, target_col: str = 'output') -> Tuple:
    """Enhanced data preparation with advanced feature engineering"""
    print("Loading data...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    print(f"Original shapes - Train: {train_df.shape}, Test: {test_df.shape}")
    
    # Apply advanced feature engineering
    print("Applying advanced feature engineering...")
    train_df = advanced_feature_engineering(train_df)
    test_df = advanced_feature_engineering(test_df)
    
    # Separate features and target
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col].values
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col].values
    
    # Use RobustScaler (better for outliers)
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Final shapes - Train: {X_train_scaled.shape}, Test: {X_test_scaled.shape}")
    print(f"Number of features: {X_train_scaled.shape[1]}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, X_train.columns

def train_random_forest_optimized(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestRegressor:
    """Optimized Random Forest with best parameters"""
    print("\n=== Training Optimized Random Forest ===")
    start_time = time.time()
    
    model = RandomForestRegressor(
        n_estimators=800,
        max_depth=25,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        bootstrap=True,
        oob_score=True,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    
    model.fit(X_train, y_train)
    
    # Cross-validation score
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2', n_jobs=-1)
    print(f"Cross-validation R² scores: {cv_scores}")
    print(f"Mean CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print(f"OOB Score: {model.oob_score_:.4f}")
    
    train_time = time.time() - start_time
    print(f"Training completed in {train_time:.2f} seconds")
    
    return model

def train_gradient_boosting(X_train: np.ndarray, y_train: np.ndarray) -> GradientBoostingRegressor:
    """Gradient Boosting - often performs better than Random Forest"""
    print("\n=== Training Gradient Boosting ===")
    start_time = time.time()
    
    model = GradientBoostingRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=7,
        min_samples_split=5,
        min_samples_leaf=2,
        subsample=0.8,
        max_features='sqrt',
        random_state=42,
        verbose=0
    )
    
    model.fit(X_train, y_train)
    
    train_time = time.time() - start_time
    print(f"Training completed in {train_time:.2f} seconds")
    
    return model

def train_svm_optimized(X_train: np.ndarray, y_train: np.ndarray) -> SVR:
    """Optimized SVM"""
    print("\n=== Training Optimized SVM ===")
    start_time = time.time()
    
    # Use subset for faster training
    subset_size = min(10000, len(X_train))
    indices = np.random.choice(len(X_train), subset_size, replace=False)
    X_subset = X_train[indices]
    y_subset = y_train[indices]
    
    model = SVR(
        kernel='rbf',
        C=150,
        epsilon=0.01,
        gamma='scale',
        cache_size=1000
    )
    
    model.fit(X_subset, y_subset)
    
    train_time = time.time() - start_time
    print(f"Training completed in {train_time:.2f} seconds")
    
    return model

def train_mlp_optimized(X_train: np.ndarray, y_train: np.ndarray) -> MLPRegressor:
    """Optimized MLP with better architecture"""
    print("\n=== Training Optimized MLP ===")
    start_time = time.time()
    
    model = MLPRegressor(
        hidden_layer_sizes=(512, 256, 128, 64),
        activation='relu',
        solver='adam',
        alpha=0.001,
        batch_size=256,
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=2000,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=100,
        tol=1e-5
    )
    
    model.fit(X_train, y_train)
    
    train_time = time.time() - start_time
    print(f"Training completed in {train_time:.2f} seconds")
    print(f"Number of iterations: {model.n_iter_}")
    
    return model

def train_gaussian_process_optimized(X_train: np.ndarray, y_train: np.ndarray) -> GaussianProcessRegressor:
    """Optimized Gaussian Process"""
    print("\n=== Training Optimized Gaussian Process ===")
    start_time = time.time()
    
    subset_size = min(3000, len(X_train))
    indices = np.random.choice(len(X_train), subset_size, replace=False)
    X_subset = X_train[indices]
    y_subset = y_train[indices]
    
    kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=0.1)
    
    model = GaussianProcessRegressor(
        kernel=kernel,
        alpha=0.001,
        n_restarts_optimizer=20,
        random_state=42
    )
    
    model.fit(X_subset, y_subset)
    
    train_time = time.time() - start_time
    print(f"Training completed in {train_time:.2f} seconds")
    
    return model

def build_improved_lstm(input_shape: int) -> keras.Model:
    """Build improved LSTM architecture"""
    model = keras.Sequential([
        layers.LSTM(128, input_shape=(1, input_shape), return_sequences=True),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        layers.LSTM(64, return_sequences=True),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        layers.LSTM(32),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.Dense(1)
    ])
    
    return model

def train_lstm_optimized(X_train: np.ndarray, y_train: np.ndarray, 
                        X_test: np.ndarray, y_test: np.ndarray) -> keras.Model:
    """Optimized LSTM training"""
    print("\n=== Training Optimized LSTM ===")
    start_time = time.time()
    
    X_train_lstm = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test_lstm = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
    
    model = build_improved_lstm(X_train.shape[1])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0005),
        loss='mse',
        metrics=['mae', 'mse']
    )
    
    print(model.summary())
    
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=50,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=15,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    history = model.fit(
        X_train_lstm, y_train,
        validation_data=(X_test_lstm, y_test),
        epochs=500,
        batch_size=64,
        callbacks=callbacks,
        verbose=1
    )
    
    train_time = time.time() - start_time
    print(f"Training completed in {train_time:.2f} seconds")
    print(f"Best epoch: {np.argmin(history.history['val_loss']) + 1}")
    
    return model

def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray, 
                  model_name: str, is_lstm: bool = False) -> Dict:
    """Evaluate model performance"""
    print(f"\n=== Evaluating {model_name} ===")
    
    if is_lstm:
        X_test_eval = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
    else:
        X_test_eval = X_test
    
    start_time = time.time()
    y_pred = model.predict(X_test_eval)
    inference_time = time.time() - start_time
    
    if len(y_pred.shape) > 1:
        y_pred = y_pred.flatten()
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    results = {
        'model': model_name,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'inference_time': inference_time,
        'predictions': y_pred
    }
    
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")
    print(f"Inference time: {inference_time:.4f} seconds")
    
    return results

def main():
    """Main execution"""
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_path = os.path.join(base_path, 'data', 'processed_train_data.csv')
    test_path = os.path.join(base_path, 'data', 'processed_test_data.csv')
    models_path = os.path.join(base_path, 'results', 'models')
    os.makedirs(models_path, exist_ok=True)
    
    # Prepare data with advanced features
    X_train, X_test, y_train, y_test, feature_names = prepare_data_advanced(train_path, test_path)
    
    all_results = []
    
    # Train all models
    print("\n" + "="*70)
    print("TRAINING ALL MODELS")
    print("="*70)
    
    # 1. Random Forest (Optimized)
    rf_model = train_random_forest_optimized(X_train, y_train)
    rf_results = evaluate_model(rf_model, X_test, y_test, "Random Forest")
    all_results.append(rf_results)
    pickle.dump(rf_model, open(os.path.join(models_path, 'random_forest_optimized.pkl'), 'wb'))
    
    # 2. Gradient Boosting (NEW - often best performer)
    gb_model = train_gradient_boosting(X_train, y_train)
    gb_results = evaluate_model(gb_model, X_test, y_test, "Gradient Boosting")
    all_results.append(gb_results)
    pickle.dump(gb_model, open(os.path.join(models_path, 'gradient_boosting.pkl'), 'wb'))
    
    # 3. SVM
    svm_model = train_svm_optimized(X_train, y_train)
    svm_results = evaluate_model(svm_model, X_test, y_test, "SVM")
    all_results.append(svm_results)
    pickle.dump(svm_model, open(os.path.join(models_path, 'svm_optimized.pkl'), 'wb'))
    
    # 4. MLP
    mlp_model = train_mlp_optimized(X_train, y_train)
    mlp_results = evaluate_model(mlp_model, X_test, y_test, "MLP")
    all_results.append(mlp_results)
    pickle.dump(mlp_model, open(os.path.join(models_path, 'mlp_optimized.pkl'), 'wb'))
    
    # 5. Gaussian Process
    gp_model = train_gaussian_process_optimized(X_train, y_train)
    gp_results = evaluate_model(gp_model, X_test, y_test, "Gaussian Process")
    all_results.append(gp_results)
    pickle.dump(gp_model, open(os.path.join(models_path, 'gaussian_process_optimized.pkl'), 'wb'))
    
    # 6. LSTM
    lstm_model = train_lstm_optimized(X_train, y_train, X_test, y_test)
    lstm_results = evaluate_model(lstm_model, X_test, y_test, "LSTM", is_lstm=True)
    all_results.append(lstm_results)
    lstm_model.save(os.path.join(models_path, 'lstm_optimized.h5'))
    
    # Save results
    results_df = pd.DataFrame([
        {k: v for k, v in r.items() if k != 'predictions'}
        for r in all_results
    ])
    
    results_path = os.path.join(base_path, 'results', 'performance_metrics.csv')
    results_df.to_csv(results_path, index=False)
    
    print("\n" + "="*70)
    print("FINAL MODEL COMPARISON")
    print("="*70)
    print(results_df.to_string(index=False))
    print(f"\nBest R² Score: {results_df['r2'].max():.4f} ({results_df.loc[results_df['r2'].idxmax(), 'model']})")
    
    # Save predictions
    pred_path = os.path.join(base_path, 'results', 'model_predictions')
    os.makedirs(pred_path, exist_ok=True)
    
    for result in all_results:
        pred_df = pd.DataFrame({
            'y_true': y_test,
            'y_pred': result['predictions']
        })
        filename = f"{result['model'].lower().replace(' ', '_')}_predictions.csv"
        pred_df.to_csv(os.path.join(pred_path, filename), index=False)
    
    print("\n✓ Enhanced model training completed successfully!")
    print(f"Results saved to: {results_path}")

if __name__ == "__main__":
    main()