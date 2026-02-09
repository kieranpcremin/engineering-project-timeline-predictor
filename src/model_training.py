"""
Model training, evaluation, and comparison.

Trains multiple regression models, evaluates with cross-validation,
and saves the best model as a joblib artifact.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

from feature_engineering import prepare_data

# Try to import XGBoost; fall back gracefully
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


def get_models() -> dict:
    """Return dictionary of model name -> model instance."""
    models = {
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, max_depth=20, min_samples_leaf=3, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.08, random_state=42
        ),
    }
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.08,
            random_state=42, n_jobs=-1, verbosity=0
        )
    return models


def evaluate_model(model, X_test_transformed, y_test) -> dict:
    """Compute evaluation metrics on the test set."""
    y_pred = model.predict(X_test_transformed)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    return {"MAE": mae, "RMSE": rmse, "R²": r2, "MAPE (%)": mape, "predictions": y_pred}


def train_and_compare(df: pd.DataFrame) -> dict:
    """Train all models, evaluate, and return results.

    Args:
        df: Raw project DataFrame.

    Returns:
        Dictionary with keys: results, best_model_name, best_model,
        preprocessor, feature_names, residual_std, y_test, y_pred_best
    """
    X_train, X_test, y_train, y_test, preprocessor, feature_names = prepare_data(df)

    X_train_t = preprocessor.transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    models = get_models()
    results = {}

    print("=" * 65)
    print("MODEL TRAINING & EVALUATION")
    print("=" * 65)

    best_r2 = -np.inf
    best_model_name = None
    best_model = None
    best_predictions = None

    for name, model in models.items():
        print(f"\n--- {name} ---")

        # 5-fold cross-validation on training set
        cv_scores = cross_val_score(
            model, X_train_t, y_train, cv=5, scoring="r2"
        )
        print(f"  CV R² scores: {cv_scores.round(3)}")
        print(f"  CV R² mean:   {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

        # Train on full training set
        model.fit(X_train_t, y_train)

        # Evaluate on test set
        metrics = evaluate_model(model, X_test_t, y_test)
        results[name] = {
            "MAE": round(metrics["MAE"], 2),
            "RMSE": round(metrics["RMSE"], 2),
            "R²": round(metrics["R²"], 4),
            "MAPE (%)": round(metrics["MAPE (%)"], 2),
            "CV R² Mean": round(cv_scores.mean(), 4),
            "CV R² Std": round(cv_scores.std(), 4),
        }

        print(f"  Test MAE:     {metrics['MAE']:.2f} weeks")
        print(f"  Test RMSE:    {metrics['RMSE']:.2f} weeks")
        print(f"  Test R²:      {metrics['R²']:.4f}")
        print(f"  Test MAPE:    {metrics['MAPE (%)']:.2f}%")

        if metrics["R²"] > best_r2:
            best_r2 = metrics["R²"]
            best_model_name = name
            best_model = model
            best_predictions = metrics["predictions"]

    # Calculate residual std for confidence intervals
    residuals = y_test.values - best_predictions
    residual_std = np.std(residuals)

    print(f"\n{'=' * 65}")
    print(f"BEST MODEL: {best_model_name} (R² = {best_r2:.4f})")
    print(f"{'=' * 65}")

    return {
        "results": results,
        "best_model_name": best_model_name,
        "best_model": best_model,
        "preprocessor": preprocessor,
        "feature_names": feature_names,
        "residual_std": residual_std,
        "y_test": y_test,
        "y_pred_best": best_predictions,
    }


def save_model(training_output: dict, save_path: Path):
    """Save the best model artifact to disk.

    Saves model, preprocessor, feature names, metrics, and metadata
    as a single joblib file.
    """
    artifact = {
        "model": training_output["best_model"],
        "preprocessor": training_output["preprocessor"],
        "feature_names": training_output["feature_names"],
        "model_name": training_output["best_model_name"],
        "metrics": training_output["results"],
        "residual_std": training_output["residual_std"],
    }
    save_path.parent.mkdir(exist_ok=True)
    joblib.dump(artifact, save_path)
    print(f"\nModel saved to {save_path}")


if __name__ == "__main__":
    data_path = Path(__file__).resolve().parent.parent / "data" / "construction_projects.csv"
    model_path = Path(__file__).resolve().parent.parent / "models" / "best_model.joblib"

    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} projects from {data_path}\n")

    output = train_and_compare(df)
    save_model(output, model_path)

    # Print comparison table
    print("\n\nMODEL COMPARISON TABLE")
    print("-" * 65)
    results_df = pd.DataFrame(output["results"]).T
    print(results_df.to_string())
