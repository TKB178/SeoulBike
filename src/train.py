"""
Training loop and model artifact serialization script using TimeSeriesSplit cross-validation.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from src.feature_engineering import engineer_all_features
from src.preprocessing import (
    build_preprocessor,
    map_binary_features,
    transform_target,
    inverse_transform_target,
)
from src.evaluation import compute_metrics


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardizes column names to handle variations in spaces and special characters."""
    df = df.copy()
    # Strip whitespace from column names
    df.columns = df.columns.str.strip()
    
    # Column mapping dictionary to standardize variations
    column_mapping = {
        'Temperature(°C)': 'Temperature (°C)',
        'Humidity(%)': 'Humidity (%)',
        'Wind speed (m/s)': 'Wind speed (m/s)',
        'Wind speed(m/s)': 'Wind speed (m/s)',
        'Visibility (10m)': 'Visibility (10m)',
        'Visibility(10m)': 'Visibility (10m)',
        'Dew point temperature(°C)': 'Dew point temperature (°C)',
        'Dew point temperature (°C)': 'Dew point temperature (°C)',
        'Solar Radiation (MJ/m2)': 'Solar Radiation (MJ/m²)',
        'Solar Radiation(MJ/m2)': 'Solar Radiation (MJ/m²)',
        'Rainfall(mm)': 'Rainfall (mm)',
        'Snowfall (cm)': 'Snowfall (cm)',
        'Snowfall(cm)': 'Snowfall (cm)',
    }
    
    return df.rename(columns=column_mapping)


def train_and_evaluate_all(data_path: str, models_dir: str = "models/"):
    os.makedirs(models_dir, exist_ok=True)
    
    # 1. Load Data
    raw_df = pd.read_csv(data_path, encoding="unicode_escape")
    
    # Clean and standardize column names first
    raw_df = standardize_columns(raw_df)
    
    # Filter functioning day for training, then preprocess
    df = engineer_all_features(raw_df)
    df = map_binary_features(df)
    
    # Separate features and target
    X = df.drop(columns=["Date", "Rented Bike Count"])
    y = df["Rented Bike Count"]
    y_log = transform_target(y)
    
    # TimeSeriesSplit cross-validation strategy
    tscv = TimeSeriesSplit(n_splits=5)
    
    models = {
        "Ridge": Ridge(),
        "RandomForest": RandomForestRegressor(random_state=42),
        "XGBoost": XGBRegressor(random_state=42),
        "CatBoost": CatBoostRegressor(verbose=0, random_state=42),
    }
    
    results = {}
    
    # Fit preprocessor on full features structure for feature names
    preprocessor = build_preprocessor()
    
    for name, model in models.items():
        print(f"Training {name} with TimeSeriesSplit CV...")
        oof_preds = np.zeros(len(df))
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y_log.iloc[train_idx], y_log.iloc[val_idx]
            
            # Preprocess splits
            X_train_proc = preprocessor.fit_transform(X_train)
            X_val_proc = preprocessor.transform(X_val)
            
            model.fit(X_train_proc, y_train)
            val_preds_log = model.predict(X_val_proc)
            
            oof_preds[val_idx] = inverse_transform_target(val_preds_log)
            
        # Compute Out-Of-Fold metrics
        eval_idx = np.concatenate([val for _, val in tscv.split(X)])
        fold_metrics = compute_metrics(y.iloc[eval_idx], oof_preds[eval_idx])
        results[name] = fold_metrics
        print(f"{name} Metrics: {fold_metrics}")
        
        # Refit on full dataset and save artifact
        X_full_proc = preprocessor.fit_transform(X)
        model.fit(X_full_proc, y_log)
        
        joblib.dump(model, os.path.join(models_dir, f"{name.lower()}.joblib"), compress = 3)        

    joblib.dump(preprocessor, os.path.join(models_dir, "preprocessor.joblib"))
    print("\n✅ All models trained and artifacts successfully saved to models/!")
    return results


if __name__ == "__main__":
    # Check if dataset exists at root or in data/raw/
    import os
    if os.path.exists("SeoulBikeData.csv"):
        data_file = "SeoulBikeData.csv"
    elif os.path.exists("data/raw/SeoulBikeData.csv"):
        data_file = "data/raw/SeoulBikeData.csv"
    else:
        data_file = "SeoulBikeData.csv"

    train_and_evaluate_all(data_file)
