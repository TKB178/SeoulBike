"""
Data preprocessing and scikit-learn transformer pipelines.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERICAL_FEATURES = [
    "Temperature (°C)",
    "Humidity (%)",
    "Wind speed (m/s)",
    "Visibility (10m)",
    "Solar Radiation (MJ/m²)",
    "Discomfort_Index",
]

CATEGORICAL_NOMINAL = ["Seasons"]

BINARY_MAPPINGS = {
    "Holiday": {"Holiday": 1, "No Holiday": 0},
    "Functioning Day": {"Yes": 1, "No": 0},
}

def build_preprocessor() -> ColumnTransformer:
    """Constructs ColumnTransformer for continuous scaling and one-hot encoding.
    
    Uses handle_unknown='ignore' so time-series validation folds missing a season
    won't throw a ValueError during transform.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERICAL_FEATURES),
            (
                "cat",
                OneHotEncoder(
                    drop=None,  # Set drop=None when using handle_unknown='ignore'
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                CATEGORICAL_NOMINAL,
            ),
        ],
        remainder="passthrough",
    )
    return preprocessor

def map_binary_features(df: pd.DataFrame) -> pd.DataFrame:
    """Converts categorical binary strings ('Holiday', 'Functioning Day') to 0/1."""
    df = df.copy()
    for col, mapping in BINARY_MAPPINGS.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)
    return df


def transform_target(y: pd.Series) -> pd.Series:
    """Log1p transform target to handle right-skewness: y_trans = log(1 + y)."""
    return np.log1p(y)


def inverse_transform_target(y_trans: np.ndarray) -> np.ndarray:
    """Inverse expm1 transform prediction back to original scale: y = exp(y_trans) - 1."""
    preds = np.expm1(y_trans)
    return np.clip(preds, 0, None)  # Prevent negative rental predictions
