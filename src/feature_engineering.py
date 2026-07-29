"""
Feature engineering module for the Seoul Bike Sharing Demand Prediction System.
Handles temporal extraction, cyclical transformations, and domain feature creation.
"""

import numpy as np
import pandas as pd


def parse_dates_and_temporal_features(df: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    """Extract Month, DayOfWeek, and IsWeekend features from raw Date column."""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], format="%d/%m/%Y")
    
    df["Month"] = df[date_col].dt.month
    df["DayOfWeek"] = df[date_col].dt.dayofweek
    df["IsWeekend"] = df["DayOfWeek"].apply(lambda x: 1 if x >= 5 else 0)
    
    return df


def add_cyclical_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """Apply sine and cosine transformations to Hour (24h) and Month (12m)."""
    df = df.copy()
    
    df["Hour_Sin"] = np.sin(2 * np.pi * df["Hour"] / 24.0)
    df["Hour_Cos"] = np.cos(2 * np.pi * df["Hour"] / 24.0)
    
    df["Month_Sin"] = np.sin(2 * np.pi * df["Month"] / 12.0)
    df["Month_Cos"] = np.cos(2 * np.pi * df["Month"] / 12.0)
    
    return df


def add_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute domain-specific indicators:
    - Discomfort Index = (1.8 * T + 32) - (0.55 - 0.0055 * RH) * (1.8 * T - 26)
    - Binary weather flags for rain and snow
    """
    df = df.copy()
    
    # Temperature (T) and Relative Humidity (RH)
    t = df["Temperature (°C)"]
    rh = df["Humidity (%)"]
    
    # Thom's Discomfort Index
    df["Discomfort_Index"] = (1.8 * t + 32) - (0.55 - 0.0055 * rh) * (1.8 * t - 26)
    
    # Binary adverse weather indicators
    df["Is_Raining"] = (df["Rainfall (mm)"] > 0).astype(int)
    df["Is_Snowing"] = (df["Snowfall (cm)"] > 0).astype(int)
    
    return df


def engineer_all_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Full feature engineering pipeline execution."""
    df = parse_dates_and_temporal_features(raw_df)
    df = add_cyclical_encoding(df)
    df = add_domain_features(df)
    return df
