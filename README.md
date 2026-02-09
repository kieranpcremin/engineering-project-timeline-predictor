# Engineering Project Timeline Predictor

**A machine learning system that predicts construction project durations from project characteristics — comparing classical ML algorithms on structured tabular data with domain-driven feature engineering.**

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5+-orange)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-green)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![R²](https://img.shields.io/badge/Best_R²-0.876-brightgreen)](#-results)

---

## What This Project Does

Input project characteristics — type, size, complexity, budget, regulatory environment — and the model predicts how many weeks the project will take to complete.

- **Regression on tabular data** — predicting continuous duration from 18 features across 2,000 synthetic construction projects
- **Multiple model comparison** — Ridge Regression, Random Forest, Gradient Boosting, XGBoost trained and evaluated side-by-side
- **Domain-driven feature engineering** — derived features encoding construction industry knowledge that boost model performance
- **Interactive demo** — 4-tab Streamlit web app with predictions, model comparison, and feature importance analysis

---

## Demo

<!-- Add screenshot here -->

---

## Results

| Model | MAE (weeks) | RMSE (weeks) | R² | MAPE (%) |
|-------|-------------|--------------|-----|----------|
| Ridge Regression | 26.4 | 34.1 | 0.858 | 9.5 |
| Random Forest | 32.0 | 40.6 | 0.799 | 12.0 |
| Gradient Boosting | 25.6 | 33.0 | 0.867 | 9.2 |
| **XGBoost** | **24.7** | **31.9** | **0.876** | **9.0** |

XGBoost wins by capturing non-linear threshold effects and feature interactions that linear models miss. Predictions are typically within **9% of actual duration** — equivalent to telling a stakeholder *"our estimates are within +/- 2 weeks on a 6-month project."*

---

## Why XGBoost Outperforms Ridge Regression

Ridge Regression can only learn additive, linear relationships — each feature gets a single fixed weight. The dataset deliberately embeds patterns that linear models cannot represent:

**1. Non-linear relationships**
Building area uses `sqrt(area)` — doubling area doesn't double duration. XGBoost learns this curve through successive splits; Ridge can only fit a straight line through it.

**2. Feature interactions**
CQV (Commissioning, Qualification & Validation) impact scales with cleanroom area and building size. A small building with 10% cleanroom and CQV adds ~1 week. A large pharma facility with 60% cleanroom adds 36+ weeks. Ridge treats CQV as a fixed +/- weeks regardless of other features.

**3. Threshold effects**
Buildings above 20,000 sqm incur a coordination penalty — a sudden step change. Decision trees capture this naturally with a single split. Ridge approximates it with a gentle slope across the whole range.

**4. Conditional logic**
Modular construction saves time on greenfield sites but has no effect on brownfield/renovation. XGBoost discovers this by splitting on facility class first, then on modular. Ridge assigns modular a single weight averaged across all facility types.

> Ridge still performs respectably (R² 0.858) because many relationships *are* roughly linear. The gap to XGBoost (0.876) represents the non-linear patterns Ridge can't capture. On real-world data with messier, more complex relationships, this gap would likely be larger.

---

## Random Forest vs Gradient Boosting

Both use hundreds of decision trees, but they build and combine them differently:

| Aspect | Random Forest | Gradient Boosting |
|--------|--------------|-------------------|
| **Strategy** | Build 300 independent trees in parallel, average predictions | Build 300 trees sequentially, each correcting the last |
| **What each tree learns** | The full target (duration) | The residual errors from previous trees |
| **Tree depth** | Deep (20) — strong individual trees | Shallow (6) — weak individual trees |
| **Main strength** | Reduces variance (overfitting) | Reduces bias (underfitting) |
| **Analogy** | Asking 300 experts and averaging | An editor revising a draft, each pass fixing remaining errors |

**Results:**

| Model | R² | MAE (weeks) |
|-------|-----|-------------|
| Random Forest | 0.799 | 32.0 |
| Gradient Boosting | 0.867 | 25.6 |
| XGBoost | 0.876 | 24.7 |

Gradient Boosting significantly outperforms Random Forest here — typical for structured tabular data. The sequential error-correction captures precise non-linear relationships and threshold effects that Random Forest's averaging tends to smooth over. XGBoost is an optimised implementation with better regularisation and efficient handling of sparse data from one-hot encoding.

---

## Feature Engineering

The model uses 18 raw features plus 5 engineered features derived from domain knowledge. This was one of the most valuable lessons: **knowing the domain matters more than knowing the algorithm.**

| Derived Feature | Formula | Why It Matters |
|----------------|---------|---------------|
| `budget_per_sqm` | budget / area | Cost intensity — EUR 100M on 50,000 sqm is a basic warehouse (EUR 2,000/sqm). EUR 100M on 2,000 sqm is a premium pharma cleanroom (EUR 50,000/sqm). The raw features don't capture this. |
| `area_per_floor` | area / floors | Distinguishes tall narrow builds from wide single-storey — different construction dynamics |
| `stakeholder_ratio` | stakeholders / team_size | Communication overhead — 15 stakeholders on a 300-person team is manageable; 15 on a 30-person team means constant interruptions |
| `is_fast_track` | 1 if design < 60% | Binary fast-track indicator — starting construction before design is complete increases risk and rework |
| `is_pharma_regulated` | 1 if FDA/EMA/both | Simplified regulatory flag — any regulated project faces validation overhead regardless of which authority |

Interaction features (`cleanroom % x CQV`, `modular x greenfield`) are deliberately left for the tree models to discover automatically — this is specifically why XGBoost outperforms Ridge Regression on this data.

---

## CQV: Why It Matters in the Model

**CQV** (Commissioning, Qualification & Validation) is a mandatory process in regulated industries — especially pharmaceutical and biotech — to prove that facilities, equipment, and systems work correctly and meet regulatory standards.

- **Commissioning** — verifying all installed systems (HVAC, utilities, process equipment) operate as designed
- **Qualification** — formal documented testing at multiple stages (IQ/OQ/PQ) to prove equipment meets specifications
- **Validation** — proving processes consistently produce results meeting predetermined criteria

In the model, CQV has a **multiplicative interaction** with cleanroom percentage and building area:

```
cqv_effect = (cleanroom% / 100) x building_area x includes_cqv x 0.002
```

This means CQV's impact on duration **scales with how much cleanroom area there is**. There's also a three-way interaction: pharma/biotech + strict regulation (FDA/both) + high cleanroom % adds an additional 35-week penalty, representing the heavy CQV burden on large regulated facilities.

### Observation: Missing Base CQV Cost

During testing, we found that enabling CQV on a pharma project with only 10% cleanroom added just ~1 week — unrealistically low. The issue: the formula has **no fixed base cost**. In reality, CQV involves a minimum overhead regardless of facility size — IQ/OQ/PQ protocols, documentation, regulatory submissions, and scheduling coordination would add at least 8–12 weeks to any CQV project.

A more realistic formula would include a fixed base cost plus a scaling component:

```python
cqv_base = includes_cqv * 10        # minimum 10 weeks for any CQV project
cqv_scaling = (cleanroom% / 100) * building_area * includes_cqv * 0.002
cqv_effect = cqv_base + cqv_scaling
```

This is an acknowledged trade-off of synthetic data — the model learns whatever the data generator encodes.

---

## Evaluation Strategy

### Cross-Validation

A single train/test split is unreliable — performance depends on *which* data points ended up in each set. **5-fold cross-validation** gives a robust estimate:

```
Fold 1: [TEST] [train] [train] [train] [train]  -> R² = 0.871
Fold 2: [train] [TEST] [train] [train] [train]  -> R² = 0.863
Fold 3: [train] [train] [TEST] [train] [train]  -> R² = 0.879
Fold 4: [train] [train] [train] [TEST] [train]  -> R² = 0.868
Fold 5: [train] [train] [train] [train] [TEST]  -> R² = 0.875
                                        Mean R² = 0.871 (+/- 0.006)
```

The low variance (+/- 0.006) confirms the model performs consistently regardless of which data it sees. CV is only performed on the **training data** (80%) — the held-out test set (20%) stays completely unseen until final evaluation, preventing data leakage.

### R² vs MAPE: Two Different Questions

| Metric | Question It Answers | XGBoost Score |
|--------|-------------------|---------------|
| **R²** | How much variation does the model explain? (1.0 = perfect) | 0.876 — explains 87.6% of variation |
| **MAPE** | How far off are predictions as a percentage? (0% = perfect) | 9.0% — predictions within 9% of actual |

R² is useful for comparing models. MAPE is useful for communicating to stakeholders: *"our estimates are typically within 9% of actual duration."*

---

## Overfitting Safeguards

With 2,000 samples and ~40+ features after one-hot encoding, overfitting is a real concern. The project employs multiple defences:

| Strategy | How It Helps |
|----------|-------------|
| **5-fold cross-validation** | Catches overfitting by testing on held-out folds — if CV scores drop significantly below training scores, the model is memorising |
| **Ridge regularisation (L2)** | Penalises large coefficients, preventing the linear model from fitting noise |
| **Tree depth limits** | `max_depth=6` for boosting prevents trees from memorising individual data points |
| **Learning rate** | XGBoost's `lr=0.08` means each tree contributes only 8%, requiring many trees to work together — no single tree dominates |
| **Separate test set** | Final evaluation on 20% of data never seen during training or CV |

---

## Architecture

```
                                ┌─────────────────┐
                                │  Streamlit App   │
                                │  (4 tabs)        │
                                └────────┬─────────┘
                                         │
                                ┌────────▼─────────┐
                                │  predictor.py     │
                                │  Load model,      │
                                │  make predictions  │
                                └────────┬─────────┘
                                         │
                ┌────────────────────────┼──────────────────────┐
                │                        │                      │
       ┌────────▼─────────┐    ┌────────▼─────────┐   ┌───────▼──────────┐
       │ data_generator.py │    │ feature_eng.py    │   │ model_training.py│
       │ 2,000 synthetic   │    │ Derived features  │   │ Train & compare  │
       │ projects          │───▶│ + preprocessing   │──▶│ 4 algorithms     │
       └──────────────────┘    └──────────────────┘   └──────────────────┘
```

---

## Known Limitations & Honest Reflection

| Limitation | Impact | What I'd Do Differently |
|-----------|--------|------------------------|
| **Synthetic data** | Model learns planted relationships, not real-world patterns. R² scores will be higher than on real data because noise is well-behaved (Gaussian). | Train on anonymised real project data |
| **No missing values** | Real data always has gaps — model hasn't learned to handle them | Add imputation strategies to the pipeline |
| **No temporal features** | Ignores season, market conditions, supply chain disruptions | Add time-series features and external data sources |
| **Static features** | Assumes all features are fixed at project start | Model feature changes over project lifetime |
| **No base CQV cost** | CQV impact on small projects is unrealistically low (~1 week) | Add fixed base cost + scaling component to data generator |

### How I'd Improve It

- **Real project data** — anonymised historical data from actual construction projects
- **SHAP values** — per-prediction explanations showing which features drive each individual prediction
- **Bayesian hyperparameter tuning** — Optuna or similar for systematic optimisation
- **Temporal features** — material price indices, labour availability, weather patterns
- **Feedback loop** — actual durations feed back into the model to improve future predictions

> The synthetic data approach was a deliberate choice to demonstrate the full ML pipeline (data, features, models, evaluation, deployment) without needing proprietary project data. The same pipeline would work on real data with minimal changes — mainly adding imputation for missing values and adjusting the feature engineering to match the actual schema.

---

## Key Concepts Demonstrated

| Concept | Where | What I Learned |
|---------|-------|---------------|
| **Regression** | `model_training.py` | Predicting continuous values vs classification — different loss functions, metrics, and evaluation |
| **Feature Engineering** | `feature_engineering.py` | Domain knowledge encoded as derived features improves model performance more than algorithm selection |
| **Model Comparison** | `model_training.py` | Training multiple algorithms and selecting the best objectively with consistent evaluation |
| **Cross-Validation** | `model_training.py` | 5-fold CV gives robust performance estimates vs a single train/test split |
| **Preprocessing Pipelines** | `feature_engineering.py` | ColumnTransformer for consistent preprocessing across train/test/production |
| **Non-Linear Relationships** | `data_generator.py` | Why tree-based models outperform linear regression on data with interactions and thresholds |

---

## Dataset

The dataset is **synthetically generated** — it does not come from real construction projects. It contains 2,000 rows, each representing a fictional construction project.

### Features (18 total)

**Categorical (7):**

| Feature | Values |
|---------|--------|
| `project_type` | pharmaceutical, biotech, data_centre, food_beverage, medical_device |
| `facility_class` | greenfield, brownfield, renovation |
| `region` | ireland, uk, mainland_europe, north_america, asia_pacific |
| `complexity_rating` | standard, complex, highly_complex |
| `regulatory_environment` | fda_regulated, ema_regulated, both, non_regulated |
| `procurement_route` | epcm, design_build, traditional, construction_management |
| `site_condition` | flat_urban, flat_rural, sloped, constrained_industrial |

**Numerical (8):**

| Feature | Range | Description |
|---------|-------|-------------|
| `budget_millions` | 5–500 | Project budget in millions |
| `building_area_sqm` | 500–50,000 | Total building area |
| `num_floors` | 1–8 | Number of floors |
| `cleanroom_percentage` | 0–80 | % of area that is cleanroom |
| `num_stakeholders` | 3–25 | Number of stakeholders |
| `team_size` | 20–500 | Project team size |
| `design_completion_pct` | 30–100 | Design completion at project start |
| `num_change_orders` | 0–40 | Number of change orders |

**Binary (3):**

| Feature | Description |
|---------|-------------|
| `includes_cqv` | Whether project includes commissioning, qualification & validation |
| `has_bim` | Whether project uses Building Information Modelling |
| `is_modular` | Whether modular construction is used |

**Target:** `duration_weeks` — generated from a formula embedding non-linear relationships, feature interactions, and threshold effects.

---

## Project Structure

```
project-timeline-predictor/
├── app/
│   └── streamlit_app.py          # Web UI (predict, compare, feature importance)
├── src/
│   ├── __init__.py
│   ├── data_generator.py         # Synthetic dataset creation (2,000 projects)
│   ├── feature_engineering.py    # Derived features + preprocessing pipeline
│   ├── model_training.py         # Train, evaluate, compare 4 models
│   └── predictor.py              # Load saved model, make predictions
├── data/                         # Generated dataset (not in repo)
├── models/                       # Saved model artifact (not in repo)
├── learningnotes.md              # Detailed learning notes
├── QandA.md                      # Technical Q&A and observations
├── requirements.txt
├── .gitignore
└── LICENSE
```

---

## Setup

### 1. Clone & Create Environment

```bash
git clone https://github.com/kieranpcremin/engineering-project-timeline-predictor.git
cd engineering-project-timeline-predictor
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux
pip install -r requirements.txt
```

### 2. Generate Data & Train Models

```bash
python src/data_generator.py
python src/model_training.py
```

This will:
- Generate 2,000 synthetic construction projects to `data/construction_projects.csv`
- Train 4 models with 5-fold cross-validation
- Save the best model to `models/best_model.joblib`

### 3. Run the Web App

```bash
streamlit run app/streamlit_app.py
```

---

## Tech Stack

| Component | Technology | Role |
|-----------|-----------|------|
| **ML Algorithms** | scikit-learn | Ridge Regression, Random Forest, Gradient Boosting, evaluation, preprocessing |
| **Gradient Boosting** | XGBoost | Industry-standard gradient boosting for tabular data |
| **Data Manipulation** | pandas, NumPy | Feature engineering and data processing |
| **Visualisation** | matplotlib, seaborn | Model comparison and feature importance charts |
| **Web App** | Streamlit | Interactive 4-tab demo with predictions and analysis |
| **Model Persistence** | joblib | Save and load trained models |

---

## Portfolio Context

This is part of a machine learning portfolio, each project covering a different data type and ML approach:

| Project | Data Type | ML Type | Key Tech |
|---------|----------|---------|----------|
| [Safety Detector](https://github.com/kieranpcremin/hard-hat-detector) | Images | Classification (CNN) | PyTorch, ResNet18 |
| [Safety Detector (.NET)](https://github.com/kieranpcremin/safety-detector-dotnet) | Images | Classification (CNN) | .NET, TensorFlow, ML.NET |
| [Semantic Search](https://github.com/kieranpcremin/semantic-search-for-technical-documents) | Text | Embeddings + Search | SentenceTransformers, FAISS |
| **Timeline Predictor** | **Tabular** | **Regression** | **scikit-learn, XGBoost** |

---

## Author

**Kieran Cremin**
Built with assistance from Claude (Anthropic)

---

## License

MIT License — Free to use, modify, and distribute.
