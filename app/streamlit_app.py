"""
streamlit_app.py - Web Demo for Project Timeline Predictor

Streamlit app with 4 tabs: Predict, Model Comparison,
Feature Importance, and About.

Run with:
    streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

# Add src directory to path so we can import our modules
sys.path.append(str(Path(__file__).parent.parent / "src"))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from predictor import TimelinePredictor
from data_generator import (
    PROJECT_TYPES, FACILITY_CLASSES, REGIONS, COMPLEXITY_RATINGS,
    REGULATORY_ENVIRONMENTS, PROCUREMENT_ROUTES, SITE_CONDITIONS,
)


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Project Timeline Predictor",
    page_icon="🏗️",
    layout="wide",
)


# =============================================================================
# LOAD MODEL (cached so it only loads once)
# =============================================================================

@st.cache_resource
def load_predictor():
    """Load the trained model. Cached to avoid reloading on every interaction."""
    model_path = Path(__file__).parent.parent / "models" / "best_model.joblib"
    if not model_path.exists():
        return None
    return TimelinePredictor(str(model_path))


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    st.title("🏗️ Project Timeline Predictor")
    st.markdown("Predict construction project durations using machine learning on project characteristics.")

    predictor = load_predictor()

    if predictor is None:
        st.error(
            "**Model not found!** Train the model first:\n\n"
            "```bash\ncd src\npython data_generator.py\npython model_training.py\n```"
        )
        return

    # Sidebar with model info
    with st.sidebar:
        st.header("Model Info")
        st.metric("Active Model", predictor.model_name)
        best_metrics = predictor.metrics.get(predictor.model_name, {})
        if best_metrics:
            st.metric("Test R²", f"{best_metrics.get('R²', 'N/A')}")
            st.metric("Test MAE", f"{best_metrics.get('MAE', 'N/A')} weeks")
        st.markdown("---")
        st.caption("Built with scikit-learn & Streamlit")

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔮 Predict Timeline",
        "📊 Model Comparison",
        "🎯 Feature Importance",
        "ℹ️ About",
    ])

    # -----------------------------------------------------------------
    # TAB 1: Predict Timeline
    # -----------------------------------------------------------------
    with tab1:
        col_input, col_result = st.columns([2, 1])

        with col_input:
            st.subheader("Project Details")

            # Row 1: Categoricals
            c1, c2, c3 = st.columns(3)
            with c1:
                project_type = st.selectbox("Project Type", PROJECT_TYPES, format_func=lambda x: x.replace("_", " ").title())
                facility_class = st.selectbox("Facility Class", FACILITY_CLASSES, format_func=lambda x: x.replace("_", " ").title())
                region = st.selectbox("Region", REGIONS, format_func=lambda x: x.replace("_", " ").title())
            with c2:
                complexity_rating = st.selectbox("Complexity", COMPLEXITY_RATINGS, format_func=lambda x: x.replace("_", " ").title())
                regulatory_environment = st.selectbox("Regulatory Environment", REGULATORY_ENVIRONMENTS, format_func=lambda x: x.replace("_", " ").title())
                procurement_route = st.selectbox("Procurement Route", PROCUREMENT_ROUTES, format_func=lambda x: x.replace("_", " ").title())
            with c3:
                site_condition = st.selectbox("Site Condition", SITE_CONDITIONS, format_func=lambda x: x.replace("_", " ").title())

            st.markdown("---")

            # Row 2: Numericals
            n1, n2, n3, n4 = st.columns(4)
            with n1:
                budget_millions = st.number_input("Budget (EUR millions)", min_value=5.0, max_value=500.0, value=50.0, step=5.0)
                building_area_sqm = st.number_input("Building Area (sqm)", min_value=500, max_value=50000, value=5000, step=500)
            with n2:
                num_floors = st.slider("Number of Floors", min_value=1, max_value=8, value=2)
                cleanroom_percentage = st.slider("Cleanroom %", min_value=0, max_value=80, value=0)
            with n3:
                num_stakeholders = st.slider("Stakeholders", min_value=3, max_value=25, value=8)
                team_size = st.number_input("Team Size", min_value=20, max_value=500, value=80, step=10)
            with n4:
                design_completion_pct = st.slider("Design Completion %", min_value=30, max_value=100, value=70)
                num_change_orders = st.slider("Change Orders", min_value=0, max_value=40, value=5)

            st.markdown("---")

            # Row 3: Binary
            b1, b2, b3 = st.columns(3)
            with b1:
                includes_cqv = st.checkbox("Includes CQV", value=False)
            with b2:
                has_bim = st.checkbox("Has BIM", value=True)
            with b3:
                is_modular = st.checkbox("Modular Construction", value=False)

        # Build prediction
        project_params = {
            "project_type": project_type,
            "facility_class": facility_class,
            "region": region,
            "complexity_rating": complexity_rating,
            "regulatory_environment": regulatory_environment,
            "procurement_route": procurement_route,
            "site_condition": site_condition,
            "budget_millions": budget_millions,
            "building_area_sqm": float(building_area_sqm),
            "num_floors": num_floors,
            "cleanroom_percentage": float(cleanroom_percentage),
            "num_stakeholders": num_stakeholders,
            "team_size": team_size,
            "design_completion_pct": float(design_completion_pct),
            "num_change_orders": num_change_orders,
            "includes_cqv": int(includes_cqv),
            "has_bim": int(has_bim),
            "is_modular": int(is_modular),
        }

        result = predictor.predict(project_params)

        with col_result:
            st.subheader("Predicted Duration")

            st.metric(
                label="Weeks",
                value=f"{result['predicted_weeks']}",
            )
            st.metric(
                label="Months",
                value=f"{result['predicted_months']}",
            )

            low = result["confidence_range"]["low_weeks"]
            high = result["confidence_range"]["high_weeks"]
            st.info(f"**Confidence range:** {low} – {high} weeks")

            st.caption(f"Model: {result['model_used']}")

            # Top factors
            st.markdown("---")
            st.subheader("Top Factors")
            fi = predictor.get_feature_importance()
            if not fi.empty:
                top5 = fi.head(5)
                for _, row in top5.iterrows():
                    name = row["feature"].replace("_", " ").title()
                    pct = row["importance"] / fi["importance"].sum() * 100
                    st.write(f"**{name}** — {pct:.1f}%")

    # -----------------------------------------------------------------
    # TAB 2: Model Comparison
    # -----------------------------------------------------------------
    with tab2:
        st.subheader("Model Comparison")

        metrics_df = pd.DataFrame(predictor.metrics).T
        metrics_df.index.name = "Model"

        # Highlight best model
        st.dataframe(
            metrics_df.style.highlight_max(subset=["R²", "CV R² Mean"], color="#c6efce")
            .highlight_min(subset=["MAE", "RMSE", "MAPE (%)"], color="#c6efce"),
            use_container_width=True,
        )

        chart1, chart2 = st.columns(2)

        with chart1:
            fig, ax = plt.subplots(figsize=(6, 4))
            models = list(predictor.metrics.keys())
            maes = [predictor.metrics[m]["MAE"] for m in models]
            colors = ["#2ecc71" if m == predictor.model_name else "#3498db" for m in models]
            ax.barh(models, maes, color=colors)
            ax.set_xlabel("MAE (weeks)")
            ax.set_title("Mean Absolute Error by Model")
            ax.invert_yaxis()
            plt.tight_layout()
            st.pyplot(fig)

        with chart2:
            fig, ax = plt.subplots(figsize=(6, 4))
            r2s = [predictor.metrics[m]["R²"] for m in models]
            ax.barh(models, r2s, color=colors)
            ax.set_xlabel("R² Score")
            ax.set_title("R² Score by Model")
            ax.set_xlim(0, 1)
            ax.invert_yaxis()
            plt.tight_layout()
            st.pyplot(fig)

    # -----------------------------------------------------------------
    # TAB 3: Feature Importance
    # -----------------------------------------------------------------
    with tab3:
        st.subheader("Feature Importance")
        fi = predictor.get_feature_importance()

        if fi.empty:
            st.warning("Feature importance not available for this model type.")
        else:
            top_n = st.slider("Number of features to show", 5, min(30, len(fi)), 15)
            top_fi = fi.head(top_n)

            fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.35)))
            ax.barh(
                top_fi["feature"].str.replace("_", " ").str.title()[::-1],
                top_fi["importance"].values[::-1],
                color="#3498db",
            )
            ax.set_xlabel("Importance")
            ax.set_title(f"Top {top_n} Feature Importances ({predictor.model_name})")
            plt.tight_layout()
            st.pyplot(fig)

            # Explanations for top features
            st.markdown("---")
            st.subheader("What Do These Features Mean?")

            explanations = {
                "project_type_food_beverage": "Food & beverage projects are baseline duration — other types are measured relative to this.",
                "project_type_data_centre": "Data centres build faster than other facility types — less regulatory overhead and simpler finishes.",
                "project_type_pharmaceutical": "Pharmaceutical facilities take ~45% longer due to cleanroom requirements, CQV, and regulatory compliance.",
                "project_type_biotech": "Biotech facilities share pharma complexity with additional process equipment requirements.",
                "project_type_medical_device": "Medical device facilities have moderate regulatory requirements, sitting between pharma and general industrial.",
                "complexity_rating_standard": "Standard complexity projects complete faster — this feature distinguishes them from complex/highly complex.",
                "complexity_rating_highly_complex": "Highly complex projects require more coordination, specialist trades, and longer commissioning.",
                "building_area_sqm": "Larger facilities need more construction time — a primary driver of project duration.",
                "cleanroom_percentage": "Cleanroom areas require specialised HVAC, finishes, and validation, significantly extending timelines.",
                "budget_millions": "Higher budgets correlate with larger scope and complexity.",
                "design_completion_pct": "Starting construction with incomplete design leads to rework and delays.",
                "num_change_orders": "Each change order disrupts workflow and extends the schedule.",
                "budget_per_sqm": "Cost intensity reflects specification complexity — pharma facilities cost more per sqm than warehouses.",
                "num_floors": "Multi-storey buildings require structural complexity and vertical logistics.",
                "team_size": "Larger teams can accelerate work, but with diminishing returns.",
                "num_stakeholders": "More stakeholders mean more coordination overhead and decision cycles.",
                "area_per_floor": "Large floor plates vs. stacked designs have different construction dynamics.",
                "stakeholder_ratio": "The ratio of stakeholders to team size indicates communication overhead.",
                "facility_class_renovation": "Renovations take longer due to existing structure constraints and phased demolition.",
            }

            for _, row in top_fi.head(8).iterrows():
                feat = row["feature"]
                if feat in explanations:
                    st.markdown(f"**{feat.replace('_', ' ').title()}**: {explanations[feat]}")

    # -----------------------------------------------------------------
    # TAB 4: About
    # -----------------------------------------------------------------
    with tab4:
        st.subheader("About This Project")
        st.markdown("""
**Project Timeline Predictor** uses machine learning to estimate construction
project durations based on project characteristics like type, size, complexity,
and regulatory requirements.

### How It Works

1. **Synthetic Data**: 2,000 realistic construction projects generated with
   domain-informed relationships between features and duration
2. **Feature Engineering**: Raw features are augmented with derived features
   (budget per sqm, scope volatility, cleanroom area) that capture domain knowledge
3. **Model Training**: Multiple algorithms (Ridge Regression, Random Forest,
   Gradient Boosting, XGBoost) are trained and compared using cross-validation
4. **Prediction**: The best model predicts duration with a confidence range

### Tech Stack

| Component | Technology |
|-----------|-----------|
| ML Models | scikit-learn, XGBoost |
| Data Processing | pandas, NumPy |
| Visualisation | matplotlib, seaborn |
| Web UI | Streamlit |
| Model Persistence | joblib |

### Known Limitations

- **Synthetic data**: Real construction projects have more variability and missing data
- **No temporal features**: Doesn't account for season, market conditions, or supply chain
- **No missing values**: Real-world data would require imputation strategies
- **Single-point features**: Doesn't capture how features change over project lifetime

### How I'd Improve It

- Use real project data (anonymised) for training
- Add time-series features (material price indices, labour availability)
- Implement Bayesian optimisation for hyperparameter tuning
- Add SHAP values for per-prediction explanations
- Build a feedback loop where actual durations improve future predictions
        """)


if __name__ == "__main__":
    main()
