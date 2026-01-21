# Predicting 30-Day Readmission with Social Determinants of Health (SDOH)

This project investigates the predictive contribution of **Social Determinants of Health (SDOH)** variables in modeling **30-day hospital readmission risk**, with a strong emphasis on **robust preprocessing**, **fair model comparison**, and **interpretability (SHAP + Bayesian Spike-and-Slab)**.

---

## Project Overview

- **Task**: Binary classification (30-day readmission: Yes / No)
- **Focus**: Assess the incremental predictive value of **10 SDOH domains** under both ML explainability and Bayesian feature selection

**Goals**:
- Quantify the marginal contribution of **10 SDOH variables** to predictive performance
- Assess variable importance using both **frequentist (XGBoost + SHAP)** and **Bayesian (Spike-and-Slab)** perspectives
- Interpret whether limited importance reflects **irrelevance** or **information redundancy** (i.e., SDOH signal being captured by other clinical/utilization features)

---

## Methodology

### Exploratory Data Analysis (EDA)
- Distributional comparisons of SDOH categories across readmission outcomes
- Visualization of outcome proportions within each SDOH domain
- Examination of missingness patterns as potentially informative signals (rather than simply discarding records)

### Predictive Modeling & Inference
- **XGBoost Classifier (binary classification)**
  - Hyperparameter tuning via cross-validation
  - Performance evaluated on a held-out test set
  - Model explainability via **SHAP** (global and local importance)

- **Bayesian Spike-and-Slab Logistic Regression**
  - Continuous spike-and-slab prior for variable selection
  - Posterior credible intervals used to assess variable importance
  - Domain-level inference by grouping one-hot encoded variables back to original SDOH domains

---

## Key Results

### Predictive Performance
- The final model achieved an **AUC of approximately 0.75**, indicating **moderate discriminative ability**.
- Adding SDOH variables did **not materially improve** overall prediction beyond existing clinical/utilization features.

### Variable Importance
- In **XGBoost + SHAP**, SDOH variables consistently ranked below core clinical/utilization features in global importance.
- In the **Spike-and-Slab framework**, only **2 out of the 10 SDOH domains** exhibited non-negligible posterior importance.

---

## Interpretation & Discussion

Importantly, **limited predictive contribution does not imply that SDOH variables are unimportant**.

Instead, the results suggest that:
- SDOH information may be **implicitly captured or proxied** by downstream clinical/utilization variables (e.g., comorbidities and care utilization patterns).
- Correlation and mediation effects can cause SDOH variables to be **statistically overshadowed**, even when they are meaningful in real-world mechanisms.
- From a modeling standpoint, **predictive dominance does not equate to causal or policy importance**.

As a result, while SDOH variables may offer limited incremental gain in a short-term prediction setting, they remain essential for:
- Risk stratification and equity-focused analysis
- Upstream intervention design
- Health system planning and population health management

