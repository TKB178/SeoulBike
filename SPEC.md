# System Specification: Seoul Bike Sharing Demand Prediction System

## 1. Executive Summary & Project Context
- **Course Code & Title:** BMDS2003 Data Science
- **Project Structure:** Group Assignment (4 Members)
- **Dataset Selected:** Seoul Bike Sharing Demand Dataset
- **Target Variable:** `Rented Bike Count` (Continuous numerical target for hourly rental demand)
- **Problem Type:** Supervised Machine Learning — Regression
- **Business Objective:** Build a robust, end-to-end data processing and machine learning solution following the **CRISP-DM framework** to forecast hourly bike demand in Seoul. The system empowers municipal transport authorities and fleet managers to optimize rebalancing operations, lower idle inventory, and improve urban mobility service availability.

---

## 2. Dataset & Business Understanding

### 2.1 Dataset Profile
- **Source:** Seoul Bike Sharing Demand Dataset (UCI Machine Learning Repository / Course Dataset 1)
- **Granularity:** Hourly records over a full calendar year (8,760 observations)
- **Primary Features:**
  - `Date`: Date of record (`dd/mm/yyyy`)
  - `Rented Bike Count`: Count of bikes rented each hour (Target)
  - `Hour`: Hour of the day (`0` to `23`)
  - `Temperature (°C)`: Continuous ambient temperature
  - `Humidity (%)`: Relative humidity percentage
  - `Wind speed (m/s)`: Wind velocity
  - `Visibility (10m)`: Atmospheric visibility distance
  - `Dew point temperature (°C)`: Dew point temperature
  - `Solar Radiation (MJ/m²)`: Solar energy intensity
  - `Rainfall (mm)`: Hourly rainfall amount
  - `Snowfall (cm)`: Hourly snowfall accumulation
  - `Seasons`: Nominal categorical (`Spring`, `Summer`, `Autumn`, `Winter`)
  - `Holiday`: Binary categorical (`Holiday` / `No Holiday`)
  - `Functioning Day`: Binary operational state (`Yes` / `No`)

### 2.2 Analytical & Business Impact
- **Operational Optimization:** Prevents station stockouts during peak commuter hours and overstocking during low-demand periods.
- **Resource Allocation:** Informs maintenance scheduling and logistics truck deployment routes.
- **Financial & Environmental Benefits:** Reduces operational expenditures by minimizing unnecessary redistribution transport while encouraging green micro-mobility adoption.

---

## 3. Data Preprocessing & Feature Engineering Pipeline

### 3.1 Temporal Feature Extraction
1. **Hour & Date Processing:** Extract `Hour` (0–23), `Month` (1–12), `DayOfWeek` (0–6), and `IsWeekend` (Binary 0/1).
2. **Cyclical Encoding:** To preserve boundary continuity (e.g., Hour 23 is adjacent to Hour 0), apply trigonometric sine/cosine transforms:
   $$\\text{Hour\\_Sin} = \\sin\\left(\\frac{2\\pi \\times \\text{Hour}}{24}\\right), \\quad \\text{Hour\\_Cos} = \\cos\\left(\\frac{2\\pi \\times \\text{Hour}}{24}\\right)$$
   $$\\text{Month\\_Sin} = \\sin\\left(\\frac{2\\pi \\times \\text{Month}}{12}\\right), \\quad \\text{Month\\_Cos} = \\cos\\left(\\frac{2\\pi \\times \\text{Month}}{12}\\right)$$

### 3.2 Engineered Domain Features
- **Discomfort / Heat Index:**
  $$\\text{Discomfort Index} = (1.8 \\times T + 32) - (0.55 - 0.0055 \\times \\text{RH}) \\times (1.8 \\times T - 26)$$
- **Adverse Weather Indicators:** Binary indicators `Is_Raining` (Rainfall > 0) and `Is_Snowing` (Snowfall > 0) to capture immediate non-linear behavioral shifts.

### 3.3 Categorical & Numerical Transformations
- **One-Hot Encoding:** Applied to `Seasons` via `OneHotEncoder(drop='first')` to avoid dummy variable traps.
- **Binary Mapping:** `Holiday` mapped to `1` (Holiday) / `0` (No Holiday); `Functioning Day` mapped to `1` (Yes) / `0` (No).
- **Target Transformation:** Logarithmic transform $y_{\\text{trans}} = \\log(1 + y)$ (`np.log1p`) applied prior to linear modeling to handle right-skewness and zero-inflation, inverted post-prediction via $\\hat{y} = \\exp(\\hat{y}_{\\text{trans}}) - 1$ (`np.expm1`).
- **Feature Scaling:** `StandardScaler` applied strictly within fold splits to continuous numerical features (Temperature, Humidity, Wind Speed, Visibility, Solar Radiation).

---

## 4. Modeling Strategy & Validation Scheme

### 4.1 Implemented Machine Learning Models (4-Member Requirement)
1. **Model 1 (Baseline Benchmark): Multiple Linear Regression with Ridge (L2) / Lasso (L1)**
   - *Rationale:* Provides a interpretable mathematical baseline. Alpha parameter tuned via grid search to penalize collinear weather predictors.
2. **Model 2: Random Forest Regressor**
   - *Rationale:* Non-parametric ensemble capable of learning non-linear feature interactions without normality assumptions.
   - *Tuning:* `n_estimators`, `max_depth`, `min_samples_leaf`, and `max_features` optimized via `RandomizedSearchCV`.
3. **Model 3: XGBoost / LightGBM Regressor**
   - *Rationale:* Gradient-boosted decision trees optimized for tabular structure, handling residual minimization sequentially.
   - *Tuning:* `learning_rate`, `max_depth`, `subsample`, `colsample_bytree`, and `reg_alpha`/`reg_lambda` regularization.
4. **Model 4: CatBoost Regressor**
   - *Rationale:* Symmetric tree architecture resistant to overfitting, specialized for numerical and categorical interactions.
   - *Tuning:* `depth`, `l2_leaf_reg`, and `iterations` tuned via Optuna Bayesian optimization.

### 4.2 Cross-Validation Strategy & Evaluation Metrics
- **Validation Scheme:** `TimeSeriesSplit(n_splits=5)` rolling-window expansion to eliminate temporal data leakage between historical training sets and future validation folds.
- **Primary Metrics:**
  - **Root Mean Squared Error (RMSE):** Heavy penalty on large operational forecasting errors.
    $$\\text{RMSE} = \\sqrt{\\frac{1}{n} \\sum_{i=1}^{n} (y_i - \\hat{y}_i)^2}$$
  - **Mean Absolute Error (MAE):** Intuitive measure of average bike count error.
  - **Coefficient of Determination ($R^2$):** Proportion of variance explained by model predictors.

---

## 5. Interactive Prototype Architecture & User Experience (UX)

### 5.1 Streamlit Module Design (`app.py`)
- **Sidebar:** Model selection dropdown, global parameter settings, and real-time environment status controls.
- **Tab 1: Real-Time Demand Predictor & Simulator**
  - Interactive sliders for Hour, Temperature, Humidity, Wind Speed, Solar Radiation, Rainfall, and Snowfall.
  - Dynamic prediction cards displaying point estimates alongside $95\\%$ confidence/prediction intervals.
  - Local model explainability integrated via Plotly-rendered SHAP Waterfall charts.
- **Tab 2: Interactive EDA & Trend Explorer**
  - Plotly weekday vs. weekend dual-line hourly demand profile.
  - Interactive weather correlation heatmaps and bivariate contour plots.
- **Tab 3: Model Evaluation & Diagnostic Suite**
  - Side-by-side leaderboard comparing Ridge, Random Forest, XGBoost, and CatBoost metrics.
  - Interactive Residuals vs. Fitted value diagnostic plots.
- **Tab 4: Operational Fleet Rebalancing Engine**
  - Actionable rebalancing recommendations based on predicted hourly demand spikes.

### 5.2 System Guardrails & Off-Duty Logic
- **`Functioning Day == No` Logic:**
  - **UI Layer:** Radio toggle selection of `"No"` disables all input weather and temporal sliders (`disabled=True`).
  - **Logic Layer:** Short-circuits model execution and forces prediction to `0` with a system warning banner.

---

## 6. Edge Cases, Risk Mitigation & Technical Tradeoffs

### 6.1 Edge Cases & Mitigation Strategies
- **Extreme Weather Anomalies:** High-intensity torrential rainfall ($> 20\\text{ mm/hr}$) or blizzards ($> 10\\text{ cm}$) clipped via `np.clip` bounds with an enforced business safety cap (demand capped at $< 50$ bikes).
- **Exogenous Shocks:** System breakdown/transit strikes monitored through continuous residual tracking to flag anomalies exceeding 3 standard deviations.

### 6.2 Model Architecture Tradeoff Matrix
| Dimension | Linear Baseline (Ridge) | Complex Ensembles (XGBoost / CatBoost) |
| :--- | :--- | :--- |
| **Predictive Performance** | Lower ($R^2 \\approx 0.55$, $\\text{RMSE} \\approx 430$) | Superior ($R^2 \\approx 0.91$, $\\text{RMSE} \\approx 180$) |
| **Inference Latency** | $< 0.1\\text{ ms}$ | $5 - 15\\text{ ms}$ |
| **Storage Footprint** | $< 10\\text{ KB}$ | $10 - 40\\text{ MB}$ |
| **Interpretability** | High (Direct algebraic coefficients) | Black-box (Requires post-hoc SHAP analysis) |
