"""
Feature engineering and preprocessing pipeline.

Creates derived features from raw project data and builds a scikit-learn
ColumnTransformer pipeline for model-ready data.
"""

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from pathlib import Path


CATEGORICAL_FEATURES = [
    "project_type",
    "facility_class",
    "region",
    "complexity_rating",
    "regulatory_environment",
    "procurement_route",
    "site_condition",
]

NUMERICAL_FEATURES = [
    "budget_millions",
    "building_area_sqm",
    "num_floors",
    "cleanroom_percentage",
    "num_stakeholders",
    "team_size",
    "design_completion_pct",
    "num_change_orders",
]

BINARY_FEATURES = [
    "includes_cqv",
    "has_bim",
    "is_modular",
]

DERIVED_FEATURES = [
    "budget_per_sqm",
    "area_per_floor",
    "stakeholder_ratio",
    "is_fast_track",
    "is_pharma_regulated",
]

TARGET = "duration_weeks"


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features to the dataset.

    Args:
        df: Raw project DataFrame.

    Returns:
        DataFrame with additional derived feature columns.
    """
    df = df.copy()

    df["budget_per_sqm"] = (df["budget_millions"] * 1e6) / df["building_area_sqm"]
    df["area_per_floor"] = df["building_area_sqm"] / df["num_floors"]
    df["stakeholder_ratio"] = df["num_stakeholders"] / df["team_size"]
    df["is_fast_track"] = (df["design_completion_pct"] < 60).astype(int)
    df["is_pharma_regulated"] = df["regulatory_environment"].isin(
        ["fda_regulated", "ema_regulated", "both"]
    ).astype(int)

    return df


def build_preprocessor(numerical_cols: list, categorical_cols: list, binary_cols: list) -> ColumnTransformer:
    """Build a scikit-learn ColumnTransformer preprocessing pipeline.

    Args:
        numerical_cols: Columns to scale with StandardScaler.
        categorical_cols: Columns to one-hot encode.
        binary_cols: Columns to pass through unchanged.

    Returns:
        Fitted-ready ColumnTransformer.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
            ("bin", "passthrough", binary_cols),
        ]
    )
    return preprocessor


def prepare_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Full data preparation: feature engineering, split, and preprocessor.

    Args:
        df: Raw project DataFrame.
        test_size: Fraction of data for testing.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test, preprocessor, feature_names)
    """
    df = add_derived_features(df)

    numerical_cols = NUMERICAL_FEATURES + [
        "budget_per_sqm",
        "area_per_floor",
        "stakeholder_ratio",
    ]
    categorical_cols = CATEGORICAL_FEATURES
    binary_cols = BINARY_FEATURES + ["is_fast_track", "is_pharma_regulated"]

    feature_cols = numerical_cols + categorical_cols + binary_cols
    X = df[feature_cols]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    preprocessor = build_preprocessor(numerical_cols, categorical_cols, binary_cols)

    # Fit on training data only
    preprocessor.fit(X_train)

    # Get feature names after transformation
    num_names = numerical_cols
    cat_names = list(preprocessor.named_transformers_["cat"].get_feature_names_out(categorical_cols))
    bin_names = binary_cols
    feature_names = num_names + cat_names + bin_names

    return X_train, X_test, y_train, y_test, preprocessor, feature_names


if __name__ == "__main__":
    data_path = Path(__file__).resolve().parent.parent / "data" / "construction_projects.csv"
    df = pd.read_csv(data_path)

    X_train, X_test, y_train, y_test, preprocessor, feature_names = prepare_data(df)

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples:     {len(X_test)}")
    print(f"Total features after preprocessing: {len(feature_names)}")
    print(f"\nFeature names:\n{feature_names}")
