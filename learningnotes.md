# Learning Notes — Project Timeline Predictor

These are my notes on what I learned building this project.

---

## 1. Why Tabular ML Fills the Portfolio Gap

My first two projects covered images (CNN for hard hat detection) and text (embeddings for semantic search). This project tackles **structured/tabular data** — the most common data type in industry.

Most real-world ML isn't flashy deep learning on images or text. It's predicting numbers from spreadsheets: sales forecasts, risk scores, project durations. Classical ML algorithms (Random Forest, XGBoost) still dominate tabular data. Deep learning rarely beats gradient boosting on structured data — this is well-established in the literature and Kaggle competitions.

Three projects, three data types:
- **Images** → CNNs, transfer learning, augmentation
- **Text** → Embeddings, similarity search, chunking
- **Tabular** → Feature engineering, model comparison, cross-validation

---

## 2. Regression vs Classification

My safety detector was a **classification** problem: hard hat or no hard hat (discrete categories). This project is **regression**: predicting a continuous number (duration in weeks).

Key differences:
- **Loss function**: Classification uses cross-entropy; regression uses mean squared error (MSE)
- **Output**: Classification gives probabilities for each class; regression gives a single number
- **Evaluation**: Classification uses accuracy, precision, recall; regression uses MAE, RMSE, R²
- **Threshold**: Classification has a decision boundary; regression doesn't

The same algorithms (Random Forest, Gradient Boosting) work for both — they just use different splitting criteria internally. scikit-learn has `RandomForestClassifier` and `RandomForestRegressor` as separate classes.

---

## 3. How the Algorithms Work

### Linear Regression (Ridge)
The simplest model: find the best straight line (hyperplane) through the data. Each feature gets a weight, and the prediction is a weighted sum plus a bias term.

```
duration = w1*area + w2*budget + w3*floors + ... + bias
```

**Ridge regularisation** adds a penalty for large weights (L2 penalty), which prevents overfitting. The `alpha` parameter controls how strong this penalty is.

**Limitation**: Can only learn linear relationships. If doubling the area doesn't double the duration (it doesn't — there are diminishing returns), linear regression misses this.

### Random Forest
An ensemble of decision trees. Each tree is trained on a random subset of the data (bagging) and considers a random subset of features at each split.

- **Why it works**: Individual trees overfit, but averaging hundreds of them smooths out the noise
- **Feature importance**: Measured by how much each feature reduces impurity across all trees
- **No scaling needed**: Trees don't care about feature magnitudes (unlike linear regression)
- **Hyperparameters**: `n_estimators` (number of trees), `max_depth` (tree depth limit)

### Gradient Boosting (and XGBoost)
Trains trees sequentially — each new tree tries to correct the errors of the previous ones. Instead of averaging independent trees (Random Forest), 
it builds a chain where each step improves on the last.

- **Learning rate**: Controls how much each tree contributes. Lower = more trees needed but better generalisation
- **XGBoost**: An optimised implementation with regularisation, parallel computation, and handling of missing values built in
- **Why it usually wins**: The sequential error-correction captures complex patterns that independent trees miss

### Why Tree Models Beat Linear Regression Here
The dataset has non-linear relationships:
- `cleanroom_percentage × includes_cqv` (interaction between features)
- `design_completion_pct` has diminishing returns (exponential, not linear)
- `is_modular` only matters on greenfield sites (conditional effect)

Linear regression can't learn these without explicit feature engineering. Trees discover them automatically by splitting on different features at different levels.

---

## 4. Feature Engineering as Domain Knowledge

This was the most valuable lesson: **knowing the domain matters more than knowing the algorithm**.

For example, `budget_per_sqm = budget / area` normalises cost by size. A EUR 100M project sounds expensive, but if it's 50,000 sqm that's EUR 2,000/sqm (a basic warehouse). 
If it's 2,000 sqm that's EUR 50,000/sqm (a premium pharma cleanroom). The raw features don't capture this — the derived feature does.

Similarly, `stakeholder_ratio = stakeholders / team_size` captures communication overhead. 15 stakeholders on a 300-person team is manageable. 15 stakeholders on a 30-person team 
means constant interruptions.

Interestingly, I deliberately kept some interaction features *out* of the engineering (like cleanroom × CQV, or modular × greenfield) to let the tree models discover them. 
This is why XGBoost outperforms Ridge Regression — the linear model can't learn interactions it isn't given, but trees find them automatically.

**Key insight**: Feature engineering is where ML meets domain expertise. But there's a balance — engineer too many features and you eliminate the advantage of non-linear models.

---

## 5. Cross-Validation Explained

A single train/test split is unreliable. If the split happens to put all the easy projects in the test set, R² looks great but doesn't reflect real performance.

**5-fold cross-validation** splits the training data into 5 parts:
1. Train on folds 1-4, test on fold 5
2. Train on folds 1-3,5, test on fold 4
3. Train on folds 1-2,4-5, test on fold 3
4. Train on folds 1,3-5, test on fold 2
5. Train on folds 2-5, test on fold 1

Each data point gets used for testing exactly once. The final score is the average across all 5 folds. This gives a much more robust estimate of how the model will perform on unseen data.

I also kept a separate held-out test set (20% of all data) that was never used during training or cross-validation. This is the final, unbiased evaluation.

**Why both?** CV helps compare models and tune hyperparameters without touching the test set. The test set is the final exam — you only look at it once.

---

## 6. Feature Importance Analysis

Tree-based models provide feature importance: how much each feature contributed to the model's predictions.

For Random Forest and Gradient Boosting, importance is based on **impurity reduction** — features that create the most useful splits across all trees get higher importance scores.

In this project, I'd expect:
- `building_area_sqm` near the top (it's the primary driver of base duration)
- `cleanroom_percentage` high (cleanrooms add massive complexity)
- `project_type` categorical features showing up (pharma has a 1.4x multiplier)
- `region` should NOT be near the top (it's not a driver in the data generation)

If `region` ranked highly, it would signal a data generation bug — region doesn't affect duration in the synthetic data, so the model shouldn't find it important.

**Limitation**: Feature importance tells you which features the model uses, not *how* it uses them. SHAP values (a future improvement) 
would show the direction and magnitude of each feature's effect on individual predictions.

---

## 7. Overfitting in Tabular ML

Overfitting is when the model memorises the training data instead of learning generalisable patterns. Signs:
- Training R² = 0.99 but test R² = 0.70
- Performance drops significantly on new data

How I guard against it:
- **Cross-validation**: Catches overfitting by testing on held-out folds
- **Regularisation**: Ridge alpha penalises large coefficients; tree max_depth limits complexity
- **Ensemble methods**: Random Forest's bagging and Gradient Boosting's learning rate both reduce overfitting
- **Separate test set**: Final evaluation on data the model has never seen

With 2,000 samples and ~40+ features (after one-hot encoding), there's enough data for the models to generalise. With fewer samples (say 50), 
overfitting would be a bigger concern and I'd need stronger regularisation.

---

## 8. Synthetic Data: Honest Limitations

I generated the data myself, which means:

**Advantages:**
- Complete control over relationships (I know the ground truth)
- No missing values, no data cleaning needed
- Can generate as much as I want
- No privacy or licensing concerns

**Limitations:**
- The model is learning relationships I planted — it can't discover anything truly new
- Real construction data has noise, outliers, missing values, and correlations I wouldn't think to include
- R² scores will be higher than on real data because the noise is well-behaved (Gaussian)
- The model won't generalise to real projects without retraining

**Why I did it anyway:** The goal is to demonstrate the ML pipeline — data processing, feature engineering, model training, evaluation, deployment. 
The same code would work on real data with minimal changes (mainly adding imputation for missing values).

Being honest about synthetic data is important. Claiming a model "predicts construction timelines" when it was trained on synthetic data would be misleading. 
The README and this document make it clear.

---

## 9. The Three Projects Together: Portfolio Story

Each project was chosen to show a different ML skill set:

| | Safety Detector | Semantic Search | Timeline Predictor |
|---|---|---|---|
| **Data type** | Images | Text | Tabular |
| **ML approach** | Deep learning (CNN) | Embeddings + similarity | Classical ML |
| **Framework** | PyTorch | SentenceTransformers + FAISS | scikit-learn + XGBoost |
| **Task** | Classification | Retrieval | Regression |
| **Key skill** | Transfer learning, augmentation | Chunking, vector search | Feature engineering, model comparison |
| **UI** | Streamlit | Streamlit | Streamlit |

Together they show:
1. I can work with different data types, not just one
2. I understand when to use deep learning vs classical ML (use classical ML for tabular data!)
3. I can go from raw data to deployed web app for each type
4. I think critically about limitations and honest evaluation

The construction/engineering domain across all three projects (safety on site, technical documents, project timelines) tells a coherent story rather than jumping between unrelated domains.
