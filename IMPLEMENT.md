# Implementation Plan: Seoul Bike Sharing Demand Prediction System

## 1. Overview & Architecture Strategy
This implementation plan translates the technical specifications in `SPEC.md` into actionable, step-by-step engineering tasks to deliver a high-performing Machine Learning solution and interactive Streamlit application adhering to the **CRISP-DM framework**.

The project deliverables consist of:
1. **Google Docs Report (Section A):** Comprehensive analysis covering Business Understanding, Preprocessing, EDA, Model Selection, Evaluation, and Strategic Business Recommendations.
2. **Code & Prototype Repository (Section B):** Structured Python codebase and interactive `app.py` Streamlit deployment.

---

## 2. Directory Structure & File Organization

The technical ZIP submission (`GroupX_RSWY1S2_DataScienceProject.zip`) will follow a modular software engineering directory structure:


```

GroupX_RSWY1S2_DataScienceProject/
│
├── data/
│   ├── raw/
│   │   └── SeoulBikeData.csv               # Raw dataset
│   └── processed/
│       ├── train_clean.csv                  # Processed training split
│       └── test_clean.csv                   # Processed testing split
│
├── notebooks/
│   ├── 01_eda_and_data_understanding.ipynb  # EDA, distributions, trend discovery
│   ├── 02_preprocessing_pipeline.ipynb      # Scaling, encodings, cyclical transforms
│   ├── 03_model_training_and_tuning.ipynb   # Model training, CV, hyperparameter search
│   └── 04_residual_and_error_analysis.ipynb # Diagnostic plots, residual analysis
│
├── src/
│   ├── **init**.py
│   ├── preprocessing.py                     # Custom transformers & ColumnTransformer
│   ├── feature_engineering.py               # Discomfort index, cyclical sin/cos features
│   ├── train.py                             # Training loop & model serialization (.pkl/.joblib)
│   └── evaluation.py                        # Metrics computation (RMSE, MAE, R²)
│
├── models/
│   ├── baseline_ridge.joblib                # Model 1: Baseline Ridge Regressor
│   ├── random_forest.joblib                 # Model 2: Random Forest Regressor
│   ├── xgboost_model.joblib                 # Model 3: XGBoost Regressor
│   └── catboost_model.joblib                # Model 4: CatBoost Regressor
│
├── app.py                                   # Streamlit Interactive Deployment Prototype
├── requirements.txt                         # Dependency manifest
├── SPEC.md                                  # System Specification
└── IMPLEMENT.md                             # Step-by-step Implementation Plan

```

---

## 3. Step-by-Step Execution Phases

### Phase 1: Environment Setup & Data Pipeline Foundation
- **Task 1.1:** Setup virtual environment (`venv` / `conda`) and create `requirements.txt` containing `pandas`, `numpy`, `scikit-learn`, `xgboost`, `catboost`, `optuna`, `matplotlib`, `seaborn`, `plotly`, `shap`, and `streamlit`.
- **Task 1.2:** Build modular feature extraction in `src/feature_engineering.py`:
  - Date parsing -> `Hour`, `Month`, `DayOfWeek`, `IsWeekend`.
  - Cyclical sine/cosine transformation for `Hour` and `Month`.
  - Engineered domain metrics: Discomfort Index, `Is_Raining`, `Is_Snowing`.
- **Task 1.3:** Build scikit-learn preprocessing pipeline in `src/preprocessing.py`:
  - Nominal feature encoding (`Seasons` via `OneHotEncoder(drop='first')`).
  - Binary feature mapping (`Holiday`, `Functioning Day`).
  - Continuous variable scaling using `StandardScaler` fitted strictly on training partitions.
  - Logarithmic target scaling (`np.log1p` / `np.expm1`).

### Phase 2: Exploratory Data Analysis (EDA) & Visualization
- **Task 2.1:** Execute full univariate summary statistics and check missing value/outlier distributions.
- **Task 2.2:** Generate publication-quality visualizations using Seaborn and Plotly:
  - **Commuter Signature:** Hourly rental demand breakdown comparing Weekday vs. Weekend curves.
  - **Seasonal & Weather Trends:** Temperature vs. Bike Demand scatter plots with lowess trendlines; Rainfall/Snowfall threshold behavior.
  - **Correlation Matrix:** Highlighting collinearity between Temperature, Dew Point, and Solar Radiation.

### Phase 3: Model Implementation & Cross-Validation
- **Task 3.1:** Implement `TimeSeriesSplit(n_splits=5)` rolling-window cross-validation scheme to prevent temporal leakage.
- **Task 3.2: Model 1 (Baseline) — Ridge/Lasso Regressor**
  - Train linear baseline; conduct `GridSearchCV` over regularization alpha values.
- **Task 3.3: Model 2 — Random Forest Regressor**
  - Fine-tune tree depth, `n_estimators`, `min_samples_leaf` using `RandomizedSearchCV`.
- **Task 3.4: Model 3 — XGBoost Regressor**
  - Tune `learning_rate`, `max_depth`, `subsample`, `colsample_bytree`, and L1/L2 regularization terms.
- **Task 3.5: Model 4 — CatBoost Regressor**
  - Implement Bayesian hyperparameter optimization using `Optuna` tuning symmetric depth and $L_2$ leaf regularization.
- **Task 3.6:** Serialize trained pipeline artifacts to `models/` folder.

### Phase 4: Model Evaluation, SHAP & Diagnostic Suite
- **Task 4.1:** Compute out-of-fold RMSE, MAE, and $R^2$ metrics across all 4 models; construct summary performance leaderboard.
- **Task 4.2:** Perform residual diagnostics: plot Residuals vs. Fitted values to inspect heteroscedasticity and zero-inflated predictions.
- **Task 4.3:** Generate global and local feature importance using SHAP (`TreeExplainer`) to explain top predictive drivers (Hour, Temperature, Solar Radiation, Rainfall).

### Phase 5: Streamlit Interactive Application (`app.py`)
- **Task 5.1 (Sidebar Controls):** Model selector dropdown, date/hour pickers, and dynamic sliders for weather metrics.
- **Task 5.2 (Tab 1 - Predictor):** Real-time point prediction with dynamic $95\%$ prediction intervals and embedded Plotly SHAP waterfall chart.
- **Task 5.3 (Tab 2 - Interactive EDA):** Embedded Plotly interactive charts for commuter dynamics and weather contour maps.
- **Task 5.4 (Tab 3 - Diagnostics):** Model metric comparison tables and live residual diagnostic plots.
- **Task 5.5 (Tab 4 - Rebalancing Engine):** Fleet maintenance alert generator based on predicted demand surges.
- **Task 5.6 (Off-Duty Guardrail):** Implement `Functioning Day == No` logic:
  - UI layer: Grey out/disable sliders when `Functioning Day = No`.
  - Logic layer: Override prediction engine and instantly output `0` with warning notification.

### Phase 6: Documentation & Final Verification
- **Task 6.1:** Draft Section A Google Docs report following CRISP-DM sections, incorporating all visual plots, academic citations (APA 7th format), and tabular model evaluations.
- **Task 6.2:** Conduct dry-run demonstration for presentation preparation (30-min group presentation).
- **Task 6.3:** Package code files into `GroupX_RSWY1S2_DataScienceProject.zip`.

---

## 4. Work Allocation Matrix (4-Member Team)

| Member Name | Primary Area | Responsibilities | Key Deliverables |
| :--- | :--- | :--- | :--- |
| **Member 1** | Data Preprocessing & EDA Lead | Data cleaning, temporal feature engineering, cyclical transforms, correlation analysis, and Seaborn/Plotly EDA visual generation. | `src/feature_engineering.py`, `notebooks/01_eda...ipynb`, Report Sections 4 & 5. |
| **Member 2** | Model Engineering Lead (Models 1 & 2) | Pipeline construction, baseline Ridge model, Random Forest tuning, TimeSeriesSplit CV strategy. | `models/baseline_ridge.joblib`, `models/random_forest.joblib`, Report Section 6. |
| **Member 3** | Advanced ML Lead (Models 3 & 4) | XGBoost, CatBoost implementation, Optuna Bayesian hyperparameter optimization, and SHAP explainability analysis. | `models/xgboost_model.joblib`, `models/catboost_model.joblib`, Report Section 7. |
| **Member 4** | Deployment & UX Lead | Streamlit interactive app (`app.py`) development, UI off-duty guardrails, fleet rebalancing engine, and repository packaging. | `app.py`, `GroupX_RSWY1S2_DataScienceProject.zip`, Report Executive Summary & Conclusion. |

---

## 5. Risk Register & Quality Assurance Checklist

| Risk / Issue | Likelihood | Impact | Mitigation Strategy |
| :--- | :--- | :--- | :--- |
| **Temporal Data Leakage** | High | High | Use `TimeSeriesSplit` rolling window cross-validation instead of standard random K-Fold. |
| **Extreme Weather Extrapolation** | Medium | Medium | Apply feature input clipping (`np.clip`) and post-inference safety caps during heavy rain ($>20\text{ mm/hr}$). |
| **Streamlit Deployment Latency** | Low | Medium | Cache trained models and preprocessing pipelines using `@st.cache_resource` in `app.py`. |
| **Collinearity in Weather Predictors** | High | Low | Apply L2 Ridge regularization and assess VIF (Variance Inflation Factor) scores. |
