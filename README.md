# Predicting 30-Day Readmission with Social Determinants of Health (SDOH)

## Overview
This project investigates the predictive contribution of **Social Determinants of Health (SDOH)** variables in modeling **30-day hospital readmission risk**. While SDOH factors are widely recognized as clinically and socially important, their **incremental predictive value** in the presence of rich clinical and utilization variables remains an open empirical question.

Using a combination of **machine learning models (XGBoost with SHAP)** and **Bayesian variable selection (Spike-and-Slab logistic regression)**, we examine whether a selected set of **10 SDOH-related variables** meaningfully improve predictive performance or exhibit independent importance.

---

## Research Question
> **Do SDOH variables provide additional predictive signal for 30-day readmission once traditional clinical and utilization features are accounted for?**

Specifically, we aim to:
- Quantify the marginal contribution of SDOH variables to predictive performance
- Assess variable importance using both frequentist and Bayesian perspectives
- Understand whether limited importance reflects irrelevance or information redundancy

---

## Data & Features
- **Outcome**:  
  - Binary indicator of 30-day readmission (`OUTCOME_BINARY`)
- **SDOH Variables (10 domains)**:
  - Alcohol use
  - Financial resource strain
  - Food insecurity
  - Housing stability
  - Physical activity
  - Safety & domestic violence
  - Social connection
  - Stress
  - Transportation
  - Utilities

Categorical SDOH variables were carefully encoded, with **missingness explicitly modeled as a category** where appropriate, to preserve information rather than discard observations.

---

## Methodology

### 1. Exploratory Data Analysis (EDA)
- Distributional comparisons of SDOH categories across readmission outcomes
- Visualization of outcome proportions within each SDOH domain
- Examination of missingness patterns as potentially informative signals

### 2. Predictive Modeling
- **XGBoost (binary classification)**
  - Hyperparameter tuning via cross-validation
  - Performance evaluated on a held-out test set
  - Model explainability via SHAP (global and local importance)

- **Bayesian Spike-and-Slab Logistic Regression**
  - Continuous spike-and-slab prior for variable selection
  - Posterior credible intervals used to assess variable importance
  - Grouped inference at the SDOH-domain level

---

## Key Results

### Predictive Performance
- The final predictive model achieved an **AUC of approximately 0.75**, indicating **moderate discriminative ability**.
- Incorporating SDOH variables did **not materially improve overall prediction accuracy** beyond existing clinical features.

### Variable Importance
- In tree-based models (XGBoost + SHAP), **SDOH variables consistently ranked below clinical/utilization variables** in global importance.
- In the **Spike-and-Slab framework**, only **2 out of the 10 SDOH domains** exhibited non-negligible posterior importance.

---

## Interpretation & Discussion

Importantly, **limited predictive contribution does not imply that SDOH variables are unimportant**.

Instead, the findings suggest that:
- SDOH information may be **implicitly captured or proxied** by downstream clinical variables (e.g., comorbidities, utilization patterns).
- High correlation and mediation effects can cause SDOH variables to be **statistically overshadowed**, even when they are causally relevant.
- From a modeling perspective, **predictive dominance does not equate to causal or policy importance**.

Thus, while SDOH variables may offer limited incremental gain in short-term prediction tasks, they remain essential for:
- Risk stratification and equity-focused analysis
- Upstream intervention design
- Health system planning and population health management

---

## Takeaways
- ✅ Predictive performance reached **AUC ≈ 0.75**
- ⚠️ SDOH variables showed **limited standalone predictive power**
- 💡 Results highlight **information redundancy rather than irrelevance**
- 📌 Emphasizes the distinction between **prediction, explanation, and policy relevance**

---

## Repository Structure
