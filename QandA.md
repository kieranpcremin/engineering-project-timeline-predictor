# Q&A Session

## Questions & Answers

### Q1: What is the data in construction_projects.csv?

The dataset is **synthetically generated** by `src/data_generator.py` — it does not come from real construction projects. It contains **2,000 rows**, 
each representing a fictional construction project, with **18 input features** and **1 target variable** (`duration_weeks`).

#### Features (18 total)

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
| `cleanroom_percentage` | 0–80 | % of area that is cleanroom (higher for pharma/biotech) |
| `num_stakeholders` | 3–25 | Number of stakeholders |
| `team_size` | 20–500 | Project team size |
| `design_completion_pct` | 30–100 | How complete the design is at project start |
| `num_change_orders` | 0–40 | Number of change orders |

**Binary (3):**

| Feature | Description |
|---------|-------------|
| `includes_cqv` | Whether project includes commissioning, qualification & validation (85% likely for pharma/biotech) |
| `has_bim` | Whether project uses Building Information Modelling (70% of projects) |
| `is_modular` | Whether modular construction is used (30% of projects) |

#### Target Variable

- `duration_weeks` — the project duration in weeks, generated from a formula that deliberately embeds **non-linear relationships** and **feature interactions** (e.g., cleanroom % x CQV, change orders x design completion, modular x greenfield). This is what makes tree-based models outperform linear models on this dataset.

#### Key Design Decisions

- **Cleanroom %** is correlated with project type (pharma/biotech get higher values) — mimicking real-world dependencies between features.
- **CQV inclusion** is also correlated with project type (85% for pharma vs 25% for others).
- Some categorical features have **non-uniform distributions** (e.g., greenfield 40%, brownfield 35%, renovation 25%).
- **Gaussian noise (~7% of duration)** is added to prevent perfect predictions.
- Duration is floored at a minimum of **8 weeks**.

### Q2: What does the feature engineering do?

The feature engineering step (`src/feature_engineering.py`) does two things: **creates new derived features** from the raw data, and **builds a preprocessing pipeline** 
to make all features model-ready.

#### 1. Derived Features

Five new features are calculated from existing columns using domain knowledge:

| Derived Feature | Formula | Why It Helps |
|----------------|---------|-------------|
| `budget_per_sqm` | `budget_millions * 1e6 / building_area_sqm` | Cost intensity — a high-spec facility (e.g. cleanroom) costs more per sqm and takes longer to build |
| `area_per_floor` | `building_area_sqm / num_floors` | Distinguishes tall narrow buildings from wide single-storey ones — different construction dynamics |
| `stakeholder_ratio` | `num_stakeholders / team_size` | Communication overhead — 20 stakeholders on a 50-person team is very different from 20 on a 500-person team |
| `is_fast_track` | `1 if design_completion_pct < 60` | Binary flag indicating fast-track construction (starting before design is complete), which increases risk |
| `is_pharma_regulated` | `1 if regulatory_environment in [fda, ema, both]` | Simplified regulatory flag — any regulated project faces validation overhead regardless of which authority |

These features encode **domain knowledge** that the raw numbers alone don't express. For example, the model can't easily learn that budget *relative to area*
matters more than budget alone — `budget_per_sqm` makes this explicit.

### Q3: How does the model training work?

The training pipeline (`src/model_training.py`) trains **four different regression models** on the same data, evaluates them all, and picks the best one.

#### Step-by-step Flow

1. **Data preparation** — calls `prepare_data()` from feature engineering, which adds derived features, splits 80/20 train/test, and fits the preprocessor on the training set.
2. **Transform** — both train and test sets are transformed through the fitted preprocessor (scaling, encoding).
3. **Train four models** — each model is trained on the same transformed training data.
4. **Evaluate each model** — using cross-validation on training data and final metrics on the held-out test set.
5. **Select the best** — the model with the highest R² on the test set wins.
6. **Save** — the best model, preprocessor, feature names, and metrics are bundled into a single `.joblib` file.

#### The Four Models

| Model | Type | Key Hyperparameters | How It Works |
|-------|------|-------------------|-------------|
| **Ridge Regression** | Linear | `alpha=1.0` | Linear regression with L2 regularisation. Finds the best straight-line (hyperplane) fit. Fast but can't capture non-linear relationships. |
| **Random Forest** | Ensemble (bagging) | `n_estimators=300, max_depth=20` | Trains 300 independent decision trees on random subsets of data and features, then averages their predictions. Reduces overfitting through diversity. |
| **Gradient Boosting** | Ensemble (boosting) | `n_estimators=300, max_depth=6, lr=0.08` | Trains 300 shallow trees sequentially — each tree corrects the errors of the previous ones. Learns complex patterns incrementally. |
| **XGBoost** | Ensemble (boosting) | `n_estimators=300, max_depth=6, lr=0.08` | Optimised gradient boosting with better regularisation, handling of sparse data, and parallel training. Industry standard for tabular data. |

#### Evaluation Strategy

**Cross-validation (on training data):**
- 5-fold CV — the training set is split into 5 parts; each model trains on 4 parts and tests on the 1 remaining, rotated 5 times.
- This gives a robust estimate of model performance and its variance (the `+/-` value).
- Scoring metric: R².

**Test set evaluation (on held-out 20%):**

| Metric | What It Measures |
|--------|-----------------|
| **MAE** (Mean Absolute Error) | Average prediction error in weeks — easy to interpret |
| **RMSE** (Root Mean Squared Error) | Like MAE but penalises large errors more heavily |
| **R²** (Coefficient of Determination) | How much variance the model explains (1.0 = perfect, 0.0 = predicting the mean) |
| **MAPE** (Mean Absolute Percentage Error) | Average error as a percentage of actual duration |

#### Why XGBoost Wins

The data generator deliberately embeds **non-linear interactions** (e.g., cleanroom % × CQV, change orders × design completion) and **threshold effects** (e.g., buildings > 20,000 sqm get a coordination penalty). 
Ridge Regression can't capture these — it only learns additive linear effects. The tree-based models (especially XGBoost) can learn splits and interactions automatically, giving them an edge.

### Q4: What is CQV in this context?

**CQV** stands for **Commissioning, Qualification, and Validation** — a mandatory process in regulated industries (especially pharma and biotech) to prove that facilities, equipment, and systems work correctly and meet regulatory standards.

- **Commissioning** — verifying that all installed systems (HVAC, utilities, process equipment) operate as designed.
- **Qualification** — formal documented testing at multiple stages (IQ/OQ/PQ: Installation, Operational, Performance Qualification) to prove equipment meets specifications.
- **Validation** — proving that processes consistently produce results meeting predetermined criteria (e.g., a cleanroom maintains the required air quality).

#### How CQV appears in the project

The `includes_cqv` feature is a binary flag (0 or 1):
- **85% of pharma/biotech/medical device** projects include CQV — because regulators (FDA, EMA) require it.
- **Only 25% of other project types** (data centres, food & beverage) include it.

#### Why CQV matters for project duration

In `data_generator.py`, CQV has a **multiplicative interaction** with cleanroom percentage and building area:

```
cqv_effect = (cleanroom_percentage / 100) * building_area_sqm * includes_cqv * 0.002
```

This means CQV's impact on duration **scales with how much cleanroom area there is**. A small building with 10% cleanroom and CQV adds little time. A large pharma facility with 60% cleanroom and CQV adds significant weeks. This is realistic — more cleanroom area means more systems to qualify and validate.

There's also a three-way interaction: pharma/biotech + strict regulation (FDA/both) + high cleanroom % adds an additional 35-week penalty, representing the heavy CQV burden on large regulated pharma facilities.

### Q5: Why does enabling CQV only add ~1 week to a pharma project with 10% cleanroom?

This is a **limitation of the synthetic data formula**. The CQV effect in `data_generator.py` is:

```python
cqv_effect = (cleanroom_percentage / 100) * building_area_sqm * includes_cqv * 0.002
```

With 10% cleanroom and a 5,000 sqm building: `(10/100) * 5000 * 1 * 0.002 = 1 week`.

The problem: there is **no base CQV cost**. The formula only adds time proportional to cleanroom area. In reality, CQV involves a minimum overhead regardless of facility size — IQ/OQ/PQ protocols, documentation, regulatory submissions, and scheduling coordination would add at least 8–12 weeks to any project that includes CQV.

A more realistic formula would include a **fixed base cost** plus a **scaling component**:

```python
# Better approach
cqv_base = includes_cqv * 10  # minimum 10 weeks for any CQV project
cqv_scaling = (cleanroom_percentage / 100) * building_area_sqm * includes_cqv * 0.002
cqv_effect = cqv_base + cqv_scaling
```

This is an acknowledged trade-off of using synthetic data — the planted relationships don't always match real-world domain knowledge perfectly. The model learns whatever patterns the data generator encodes, so if the generator underestimates CQV impact on small projects, the model will too.

### Q6: What does cross-validation do and why is it important?

#### The Problem It Solves

With a single train/test split, your performance score depends on *which* data points ended up in the test set. You might get lucky (easy test samples) or unlucky (hard ones). A single score doesn't tell you how **reliable** the model is.

#### How 5-Fold Cross-Validation Works (as used in this project)

The training data is split into 5 equal parts ("folds"):

```
Fold 1: [TEST] [train] [train] [train] [train]  → R² = 0.871
Fold 2: [train] [TEST] [train] [train] [train]  → R² = 0.863
Fold 3: [train] [train] [TEST] [train] [train]  → R² = 0.879
Fold 4: [train] [train] [train] [TEST] [train]  → R² = 0.868
Fold 5: [train] [train] [train] [train] [TEST]  → R² = 0.875
                                        Mean R² = 0.871 (+/- 0.006)
```

Each fold takes a turn being the test set while the other 4 are used for training. This gives you **5 scores instead of 1**.

#### Why It Matters

| Benefit | Explanation |
|---------|-------------|
| **Reliable estimate** | The mean of 5 scores is more trustworthy than a single score |
| **Variance check** | The `+/-` value tells you how stable the model is. A low variance (e.g. +/- 0.006) means it performs consistently. A high variance (e.g. +/- 0.05) means it's sensitive to which data it sees |
| **Overfitting detection** | If CV scores are much lower than training scores, the model is memorising rather than learning |
| **Fair model comparison** | Comparing models on a single split can be misleading — CV gives a fairer comparison |

#### How It's Used in This Project

In `model_training.py` (line 87–91), each model gets 5-fold CV on the training set:

```python
cv_scores = cross_val_score(model, X_train_t, y_train, cv=5, scoring="r2")
```

This happens **before** the final test set evaluation. The CV score tells you how the model is likely to perform; the held-out test set gives the final unbiased score.

#### Important Detail

CV is only done on the **training data** (80%). The test set (20%) is never touched during CV — it stays completely unseen until final evaluation. This prevents data leakage.

### Q7: What is the difference between R² and MAPE?

They both measure model performance but answer **different questions**.

#### R² (Coefficient of Determination)

**Question it answers:** "How much of the variation in project duration does the model explain?"

- **Scale:** 0.0 to 1.0 (can go negative for very bad models)
- **1.0** = perfect — the model explains all variation
- **0.0** = the model is no better than just predicting the average duration every time
- **0.876** (XGBoost in this project) = the model explains 87.6% of the variation in duration

R² is **relative** — it compares the model to a naive baseline (predicting the mean). It tells you nothing about the actual size of the errors in weeks.

#### MAPE (Mean Absolute Percentage Error)

**Question it answers:** "On average, how far off are predictions as a percentage of the actual value?"

- **Scale:** 0% upwards (lower is better)
- **0%** = perfect predictions
- **9.0%** (XGBoost in this project) = predictions are on average 9% off from actual duration

MAPE is **interpretable in business terms** — you can tell a stakeholder "our estimates are typically within 9% of actual duration."

#### Practical Example

| Project | Actual | Predicted | Absolute Error | Percentage Error |
|---------|--------|-----------|---------------|-----------------|
| Small project | 50 weeks | 55 weeks | 5 weeks | 10% |
| Large project | 300 weeks | 285 weeks | 15 weeks | 5% |

- **MAE** would say the large project had a bigger error (15 > 5 weeks)
- **MAPE** would say the small project had a bigger error (10% > 5%)
- **R²** wouldn't tell you about either individual project — it measures overall pattern fit

#### Why Use Both?

| Metric | Strength | Weakness |
|--------|----------|----------|
| **R²** | Good for comparing models against each other and against the naive baseline | Doesn't tell you error size in real units |
| **MAPE** | Easy to communicate to non-technical stakeholders ("within 9%") | Can be misleading for very short projects (5 weeks off on a 10-week project = 50% MAPE) |

In this project, XGBoost scores R² = 0.876 and MAPE = 9.0%. Together they tell you: the model captures most of the pattern in the data, and its predictions are typically within 9% of reality.

### Q8: Why does XGBoost outperform Ridge Regression on this data?

It comes down to the **shape of the relationships** in the data. Ridge can only learn straight lines; XGBoost can learn curves, thresholds, and interactions.

#### What Ridge Regression Can Do

Ridge learns a formula like:

```
duration = w1 × budget + w2 × area + w3 × cleanroom% + ... + bias
```

Each feature gets a single weight, and the effects are **additive and linear**. If increasing cleanroom from 10% to 20% adds 5 weeks, then 20% to 30% also adds exactly 5 weeks. Ridge has no way to represent "it depends."

#### What XGBoost Can Do That Ridge Can't

XGBoost builds decision trees that can capture:

**1. Non-linear relationships (curves and diminishing returns)**

The data generator uses `np.sqrt(building_area_sqm)` — doubling area doesn't double duration. XGBoost can learn this curve through successive splits; Ridge can only fit a straight line through it.

**2. Feature interactions (the effect of X depends on Y)**

```python
# CQV effect depends on cleanroom % AND building area
cqv_effect = (cleanroom_percentage / 100) * building_area_sqm * includes_cqv * 0.002
```

CQV adds almost nothing on a small low-cleanroom project but adds 36 weeks on a large high-cleanroom one. Ridge treats CQV as a fixed +/- weeks regardless of other features. XGBoost can split on CQV *within* a branch that already split on cleanroom %, effectively learning the interaction.

**3. Threshold effects (step changes)**

```python
# Buildings > 20,000 sqm get a coordination penalty
large_building_penalty = np.where(building_area_sqm > 20000, 25 + ..., 0)
```

Below 20,000 sqm: no penalty. Above: a sudden jump. Decision trees capture this naturally with a single split. Ridge has to approximate it with a gentle slope across the whole range.

**4. Conditional logic (A only matters when B is true)**

```python
# Modular construction only helps on greenfield sites
modular_benefit = is_modular * is_greenfield * np.sqrt(building_area_sqm) * (-0.1)
```

Modular construction saves time on greenfield but does nothing on brownfield/renovation. XGBoost can learn this by splitting on facility class first, then on modular. Ridge gives modular a single weight that averages across all facility types.

#### The Results Tell the Story

| Model | R² | MAE (weeks) |
|-------|-----|-------------|
| Ridge Regression | 0.858 | 26.4 |
| XGBoost | 0.876 | 24.7 |

Ridge actually does respectably well (0.858) because many of the relationships *are* roughly linear. The gap (0.858 vs 0.876) represents the non-linear interactions and thresholds that Ridge can't capture. On real-world data with messier, more complex relationships, this gap would likely be larger.

#### Why Not Just Always Use XGBoost?

Ridge has advantages too: it's faster to train, fully interpretable (each feature has one coefficient), and less prone to overfitting on small datasets. XGBoost wins here because the data has enough complexity and enough samples (2,000) to reward its flexibility.

### Q9: What is the difference between Random Forest and Gradient Boosting?

Both use **many decision trees** to make predictions, but they build and combine those trees in fundamentally different ways.

#### Random Forest (Bagging)

**Strategy: Build many independent trees in parallel, then average their answers.**

```
Tree 1 (random subset of data + features) → 130 weeks
Tree 2 (different random subset)           → 142 weeks
Tree 3 (different random subset)           → 128 weeks
...
Tree 300                                   → 135 weeks
                                    Average → 134 weeks
```

- Each tree is trained on a **random sample** of the data (with replacement) and a **random subset of features**.
- Trees are deep (up to `max_depth=20` in this project) — each individual tree is a strong but noisy predictor.
- Trees don't know about each other — they're built independently.
- The averaging cancels out individual trees' mistakes, reducing **variance** (overfitting).

**Analogy:** Asking 300 independent experts for their estimate and averaging the answers. Each expert sees slightly different information, so their individual biases cancel out.

#### Gradient Boosting

**Strategy: Build trees sequentially, where each new tree corrects the mistakes of all previous trees.**

```
Tree 1: predicts duration                    → 134 weeks (actual: 150)
         residual error = 16 weeks
Tree 2: predicts the 16-week ERROR           → corrects by +12 weeks
         remaining error = 4 weeks
Tree 3: predicts the 4-week ERROR            → corrects by +3 weeks
         remaining error = 1 week
...
Tree 300: fine-tunes the last small errors
                              Final answer → 149.5 weeks
```

- Each tree is **shallow** (`max_depth=6`) — a weak predictor on its own.
- Each new tree is trained on the **residual errors** of the ensemble so far.
- Trees are added with a **learning rate** (0.08) that controls how much each tree contributes — smaller steps = more robust but needs more trees.
- This gradually reduces **bias** (underfitting).

**Analogy:** An editor revising a draft. Each pass fixes the biggest remaining errors, with each revision making smaller and smaller corrections.

#### Side-by-Side Comparison

| Aspect | Random Forest | Gradient Boosting |
|--------|--------------|-------------------|
| **How trees are built** | Independently, in parallel | Sequentially, each correcting the last |
| **What each tree learns** | The full target (duration) | The residual errors from previous trees |
| **Tree depth** | Deep (20) — strong individual trees | Shallow (6) — weak individual trees |
| **Combination method** | Average all predictions | Sum all corrections |
| **Main strength** | Reduces variance (overfitting) | Reduces bias (underfitting) |
| **Overfitting risk** | Lower — hard to overfit by averaging | Higher — can overfit if too many trees or learning rate too high |
| **Training speed** | Faster (parallelisable) | Slower (sequential) |
| **Tuning difficulty** | Easier — fewer sensitive hyperparameters | Harder — learning rate, n_estimators, and depth all interact |

#### Results in This Project

| Model | R² | MAE (weeks) |
|-------|-----|-------------|
| Random Forest | 0.799 | 32.0 |
| Gradient Boosting | 0.867 | 25.6 |
| XGBoost | 0.876 | 24.7 |

Gradient Boosting significantly outperforms Random Forest here. This is typical for structured tabular data — the sequential error-correction approach is better at learning the precise non-linear relationships and threshold effects in this dataset. Random Forest's averaging tends to smooth over sharp transitions (like the 20,000 sqm penalty).

XGBoost is an optimised version of Gradient Boosting with better regularisation, efficient handling of sparse data (from one-hot encoding), and built-in parallelism — which is why it edges ahead.

