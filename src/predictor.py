"""
Prediction interface for the trained timeline model.

Loads a saved model artifact and provides a clean API for making
predictions from project parameters.
"""

import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from feature_engineering import add_derived_features, CATEGORICAL_FEATURES, NUMERICAL_FEATURES, BINARY_FEATURES


class TimelinePredictor:
    """Load a trained model and make timeline predictions."""

    def __init__(self, model_path: str):
        """Load the saved model artifact.

        Args:
            model_path: Path to the .joblib model file.
        """
        artifact = joblib.load(model_path)
        self.model = artifact["model"]
        self.preprocessor = artifact["preprocessor"]
        self.feature_names = artifact["feature_names"]
        self.model_name = artifact["model_name"]
        self.metrics = artifact["metrics"]
        self.residual_std = artifact["residual_std"]

    def predict(self, project_params: dict) -> dict:
        """Predict project duration from parameters.

        Args:
            project_params: Dictionary of project features matching the
                training schema.

        Returns:
            Dictionary with predicted_weeks, predicted_months,
            confidence_range, and model_used.
        """
        df = pd.DataFrame([project_params])
        df = add_derived_features(df)

        # Build feature columns in the same order as training
        numerical_cols = NUMERICAL_FEATURES + [
            "budget_per_sqm",
            "area_per_floor",
            "stakeholder_ratio",
        ]
        categorical_cols = CATEGORICAL_FEATURES
        binary_cols = BINARY_FEATURES + ["is_fast_track", "is_pharma_regulated"]

        feature_cols = numerical_cols + categorical_cols + binary_cols
        X = df[feature_cols]

        X_transformed = self.preprocessor.transform(X)
        predicted_weeks = float(self.model.predict(X_transformed)[0])
        predicted_weeks = max(predicted_weeks, 4.0)

        return {
            "predicted_weeks": round(predicted_weeks, 1),
            "predicted_months": round(predicted_weeks / 4.33, 1),
            "confidence_range": {
                "low_weeks": round(max(predicted_weeks - self.residual_std, 4.0), 1),
                "high_weeks": round(predicted_weeks + self.residual_std, 1),
            },
            "model_used": self.model_name,
        }

    def get_feature_importance(self) -> pd.DataFrame:
        """Get ranked feature importances from the trained model.

        Returns:
            DataFrame with feature and importance columns, sorted descending.
        """
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
        elif hasattr(self.model, "coef_"):
            importances = np.abs(self.model.coef_)
        else:
            return pd.DataFrame(columns=["feature", "importance"])

        fi_df = pd.DataFrame({
            "feature": self.feature_names,
            "importance": importances,
        })
        fi_df = fi_df.sort_values("importance", ascending=False).reset_index(drop=True)
        return fi_df


if __name__ == "__main__":
    model_path = Path(__file__).resolve().parent.parent / "models" / "best_model.joblib"
    predictor = TimelinePredictor(str(model_path))

    sample_project = {
        "project_type": "pharmaceutical",
        "facility_class": "greenfield",
        "region": "ireland",
        "complexity_rating": "complex",
        "regulatory_environment": "both",
        "procurement_route": "epcm",
        "site_condition": "flat_urban",
        "budget_millions": 120.0,
        "building_area_sqm": 15000.0,
        "num_floors": 3,
        "cleanroom_percentage": 45.0,
        "num_stakeholders": 12,
        "team_size": 150,
        "design_completion_pct": 70.0,
        "num_change_orders": 8,
        "includes_cqv": 1,
        "has_bim": 1,
        "is_modular": 0,
    }

    result = predictor.predict(sample_project)
    print(f"Model: {result['model_used']}")
    print(f"Predicted duration: {result['predicted_weeks']} weeks ({result['predicted_months']} months)")
    print(f"Confidence range: {result['confidence_range']['low_weeks']}-{result['confidence_range']['high_weeks']} weeks")

    print(f"\nTop 10 features:")
    fi = predictor.get_feature_importance()
    print(fi.head(10).to_string(index=False))
