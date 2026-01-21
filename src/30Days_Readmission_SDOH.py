import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, roc_auc_score
from sklearn.ensemble import RandomForestClassifier as RFC
from xgboost import XGBClassifier as XGBC
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import shap

data = pd.read_csv("readmit_with_sdoh_DEID.csv")
origin_feature_name = data.drop(columns = ["OUTCOME_BINARY"]).columns

# region ========== Feature Engineering ==========
# Make all NaN as a category (Blue in this case)
data["VAR_SDOHAlcoholUseDomainRisk_CAT"] = data["VAR_SDOHAlcoholUseDomainRisk_CAT"].apply(lambda x : "Blue" if pd.isnull(x) else x)
data["VAR_PhysicalActivityDomain_CAT"] = data["VAR_PhysicalActivityDomain_CAT"].apply(lambda x : "Blue" if pd.isnull(x) else x)
data["VAR_SDOHSocialConnectionDomain_CAT"] = data["VAR_SDOHSocialConnectionDomain_CAT"].apply(lambda x : "Blue" if pd.isnull(x) else x)
data["VAR_StressDomain_CAT"] = data["VAR_StressDomain_CAT"].apply(lambda x : "Blue" if pd.isnull(x) else x)

# Special Operation
# data["VAR_LOS_CAT"] = data["VAR_LOS_CAT"].apply(lambda x : 10 if x == "10+" else x).astype("int64")
# data["VAR_ComorbidityIndex_CAT"] = data["VAR_ComorbidityIndex_CAT"].apply(lambda x : 10 if x == "10+" else x).astype("int64")
# data["VAR_LACE_CAT"] = data["VAR_LACE_CAT"].apply(lambda x : 14 if x == "14+" else x).astype("int64")
# endregion

# region ========== EDA ==========
def distributionForDiffCategory(list_of_category):
    # Split the variable based on the category
    grouped = data.groupby([list_of_category, 'OUTCOME_BINARY']).size().unstack(fill_value=0)
    normalized = grouped.div(grouped.sum(axis=1), axis=0)

    ax = normalized.plot(
        kind='bar',
        stacked=True,
        figsize=(8, 6),
        color=["skyblue", "orange"]
    )
    ax.set_xticks(range(len(normalized.index)))
    ax.set_xticklabels([str(v) for v in normalized.index], rotation=0)

    # Split and construct the name of the feature
    name = list_of_category
    middle_name = name.split("_")[1].split("Domain")[0]
    if 'SDOH' in middle_name:
        middle_name = middle_name.split("SDOH")[1]
    res_name = ""

    for i, char in enumerate(middle_name):
        if char.isupper() and i != 0:
            res_name += " " + char
        else:
            res_name += char

    file_name = 'Outcome vs ' + res_name + '.pdf'
    # Plot
    plt.title(res_name,fontsize = 16)
    plt.ylabel('Outcome Proportion')
    plt.xlabel("")
    plt.legend(labels=["Not Readmitted", "Readmitted"], title='Readmission', loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xticks(rotation=0)
    plt.subplots_adjust(left=0.1, right=0.75)
    plt.savefig(file_name, format='pdf')
    plt.show()

def distributionForDiffOutcome(feature_col, sort_feature_categories=True, save_pdf=True):
    grouped = data.groupby(["OUTCOME_BINARY", feature_col]).size().unstack(fill_value=0)
    normalized = grouped.div(grouped.sum(axis=1), axis=0)

    if sort_feature_categories:
        category_order = normalized.sum(axis=0).sort_values(ascending=False).index.tolist()
        normalized = normalized[category_order]

    palette = ["lightblue", "lightgreen", "salmon", "plum"]
    colors = palette[:normalized.shape[1]]

    ax = normalized.plot(
        kind="bar",
        stacked=True,
        figsize=(8, 6),
        color=colors
    )

    ax.set_xticks(range(len(normalized.index)))
    ax.set_xticklabels(list(("Not Readmitted", "Readmitted")), rotation=0)

    name = feature_col
    try:
        middle_name = name.split("_")[1].split("Domain")[0]
        if "SDOH" in middle_name:
            middle_name = middle_name.split("SDOH")[1]
    except Exception:
        middle_name = name

    res_name = ""
    for i, char in enumerate(str(middle_name)):
        if char.isupper() and i != 0:
            res_name += " " + char
        else:
            res_name += char

    plt.title(f"{res_name} categories distribution")
    plt.ylabel("Proportion")
    plt.xlabel("Readmission")
    plt.legend(title=res_name, loc="center left", bbox_to_anchor=(1, 0.5))
    plt.xticks(rotation=0)
    plt.subplots_adjust(left=0.15, right=0.75)

    if save_pdf:
        file_name = f"{res_name} distribution within outcome.pdf"
        plt.savefig(file_name, format="pdf", bbox_inches="tight")

    plt.show()

SDOH_Var = ['VAR_SDOHAlcoholUseDomainRisk_CAT',
            'VAR_FinancialResourceStrainDomainCollected_FLG',
            'VAR_FoodInsecurityDomainCollected_FLG',
            'VAR_HousingStabilityDomainCollected_FLG',
            'VAR_PhysicalActivityDomain_CAT',
            'VAR_SafetyandDomesticViolenceDomainCollected_FLG',
            'VAR_SDOHSocialConnectionDomain_CAT',
            'VAR_StressDomain_CAT',
            'VAR_TransportationDomainCollected_FLG',
            'VAR_UtilitiesDomain_FLG']

for i in range(len(SDOH_Var)):
    distributionForDiffCategory(SDOH_Var[i])
    distributionForDiffOutcome(SDOH_Var[i])

# endregion

# region ========== Feature Engineering ==========
# Create categorical variables (0/1) for all sdoh variables for each category(To see the importance of nan(Blue))
data = pd.get_dummies(data, columns=["VAR_SDOHAlcoholUseDomainRisk_CAT",
                                     "VAR_PhysicalActivityDomain_CAT",
                                     "VAR_SDOHSocialConnectionDomain_CAT",
                                     "VAR_StressDomain_CAT"], drop_first=False)

data = pd.get_dummies(data, drop_first=True)
data = data.rename(columns={"VAR_Age_CAT_<20": "VAR_Age_CAT_20",
                            "VAR_Age_CAT_80+": "VAR_Age_CAT_80"})

data = data.astype("int64")
# endregion

# train, test = train_test_split(data, test_size=0.5, random_state=42)
# train.to_csv("Training_50%.csv", index=False)
# test.to_csv("Testing_50%.csv", index=False)

train = pd.read_csv('Training_50%.csv')
test = pd.read_csv('Testing_50%.csv')

# region ========== Test for effect of NA values ==========
# Remove all NA values
# train = train[~(train["VAR_SDOHAlcoholUseDomainRisk_CAT_Blue"] == 1)]
# train = train[~(train["VAR_PhysicalActivityDomain_CAT_Blue"] == 1)]
# train = train[~(train["VAR_SDOHSocialConnectionDomain_CAT_Blue"] == 1)]
# train = train[~(train["VAR_StressDomain_CAT_Blue"] == 1)]
#
# test = test[~(test["VAR_SDOHAlcoholUseDomainRisk_CAT_Blue"] == 1)]
# test = test[~(test["VAR_PhysicalActivityDomain_CAT_Blue"] == 1)]
# test = test[~(test["VAR_SDOHSocialConnectionDomain_CAT_Blue"] == 1)]
# test = test[~(test["VAR_StressDomain_CAT_Blue"] == 1)]
#
# train = train.drop(columns=["VAR_SDOHAlcoholUseDomainRisk_CAT_Blue",
#                             "VAR_PhysicalActivityDomain_CAT_Blue",
#                             "VAR_SDOHSocialConnectionDomain_CAT_Blue",
#                             "VAR_StressDomain_CAT_Blue"])
# test = test.drop(columns=["VAR_SDOHAlcoholUseDomainRisk_CAT_Blue",
#                           "VAR_PhysicalActivityDomain_CAT_Blue",
#                           "VAR_SDOHSocialConnectionDomain_CAT_Blue",
#                           "VAR_StressDomain_CAT_Blue"])

# Remove all variable with NA values
# train = train.drop(columns=["VAR_SDOHAlcoholUseDomainRisk_CAT_Blue",
#                             "VAR_SDOHAlcoholUseDomainRisk_CAT_Green",
#                             "VAR_SDOHAlcoholUseDomainRisk_CAT_Red",
#                             "VAR_PhysicalActivityDomain_CAT_Blue",
#                             "VAR_PhysicalActivityDomain_CAT_Green",
#                             "VAR_PhysicalActivityDomain_CAT_Orange",
#                             "VAR_PhysicalActivityDomain_CAT_Red",
#                             "VAR_SDOHSocialConnectionDomain_CAT_Blue",
#                             "VAR_SDOHSocialConnectionDomain_CAT_Green",
#                             "VAR_SDOHSocialConnectionDomain_CAT_Orange",
#                             "VAR_SDOHSocialConnectionDomain_CAT_Red",
#                             "VAR_StressDomain_CAT_Blue",
#                             "VAR_StressDomain_CAT_Green",
#                             "VAR_StressDomain_CAT_Red"])
# test = test.drop(columns=["VAR_SDOHAlcoholUseDomainRisk_CAT_Blue",
#                             "VAR_SDOHAlcoholUseDomainRisk_CAT_Green",
#                             "VAR_SDOHAlcoholUseDomainRisk_CAT_Red",
#                             "VAR_PhysicalActivityDomain_CAT_Blue",
#                             "VAR_PhysicalActivityDomain_CAT_Green",
#                             "VAR_PhysicalActivityDomain_CAT_Orange",
#                             "VAR_PhysicalActivityDomain_CAT_Red",
#                             "VAR_SDOHSocialConnectionDomain_CAT_Blue",
#                             "VAR_SDOHSocialConnectionDomain_CAT_Green",
#                             "VAR_SDOHSocialConnectionDomain_CAT_Orange",
#                             "VAR_SDOHSocialConnectionDomain_CAT_Red",
#                             "VAR_StressDomain_CAT_Blue",
#                             "VAR_StressDomain_CAT_Green",
#                             "VAR_StressDomain_CAT_Red"])
# train = train.drop(columns=["VAR_FinancialResourceStrainDomainCollected_FLG",
#                             "VAR_FoodInsecurityDomainCollected_FLG",
#                             "VAR_HousingStabilityDomainCollected_FLG",
#                             "VAR_SafetyandDomesticViolenceDomainCollected_FLG",
#                             "VAR_TransportationDomainCollected_FLG",
#                             "VAR_UtilitiesDomain_FLG"])
# test = test.drop(columns=["VAR_FinancialResourceStrainDomainCollected_FLG",
#                           "VAR_FoodInsecurityDomainCollected_FLG",
#                           "VAR_HousingStabilityDomainCollected_FLG",
#                           "VAR_SafetyandDomesticViolenceDomainCollected_FLG",
#                           "VAR_TransportationDomainCollected_FLG",
#                           "VAR_UtilitiesDomain_FLG"])
# endregion

# region ========== XGBOOST ==========
X_train = train.drop(columns=['OUTCOME_BINARY'])
y_train = train['OUTCOME_BINARY']
X_test = test.drop(columns=['OUTCOME_BINARY'])
y_test = test['OUTCOME_BINARY']

X_train = X_train.drop(columns=['VAR_LACE_CAT'])
X_test = X_test.drop(columns=['VAR_LACE_CAT'])

print(len(data.columns))

xgbc = XGBC(random_state=42, use_label_encoder=False, eval_metric='logloss')
param_grid_xgb = {'n_estimators': [50, 100, 200],
              'learning_rate': [0.01, 0.1, 0.2],
              'max_depth': [3, 5, 7],
              'subsample': [0.8, 1],
              'scale_pos_weight': [4],
              'eval_metric': ['logloss']}
grid_search_xgb = GridSearchCV(xgbc,
                               param_grid_xgb,
                               scoring='f1',
                               cv=10,
                               n_jobs=-1)
grid_search_xgb.fit(X_train, y_train)
print(f"Best Parameters: {grid_search_xgb.best_params_}")
best_xgb = grid_search_xgb.best_estimator_
y_pred_best_xgb = best_xgb.predict(X_test)
conf_matrix_best_xgb_test = confusion_matrix(y_test, y_pred_best_xgb)
print("Test Confusion Matrix:")
print(conf_matrix_best_xgb_test)
y_pred_best_xgb_train = best_xgb.predict(X_train)
conf_matrix_best_xgb_train = confusion_matrix(y_train, y_pred_best_xgb_train)
print("Train Confusion Matrix:")
print(conf_matrix_best_xgb_train)

# Get confusion matrix at different threshold
y_prob = best_xgb.predict_proba(X_test)[:, 1]
for i in range(1,5):
    threshold = i * 0.1
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    print(f"Test Confusion Matrix with threshold = {threshold}:")
    print(cm)

# ROC Curve
y_prob = best_xgb.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc_score = roc_auc_score(y_test, y_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.2f})', linewidth=2)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate (FPR)', fontsize=16)
plt.ylabel('True Positive Rate (TPR)', fontsize=16)
plt.title('ROC Curve', fontsize = 16)
plt.legend(loc='lower right')
plt.grid()
plt.savefig("ROC_Reduced.pdf", format = "pdf")
plt.show()

# Importance
feature_importance = best_xgb.get_booster().get_score(importance_type='gain')
sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)

# Reunite the splitted categorical variables
origin_importance = {}
for name in origin_feature_name:
    new_feature = name
    new_importance = 0
    for feature, importance in feature_importance.items():
        if feature == name or feature.startswith(name + "_"):
            new_importance += importance
    temp = {new_feature: new_importance}
    origin_importance.update(temp)
sorted_origin_importance = sorted(origin_importance.items(), key=lambda x: x[1], reverse=True)

# Combine the top 10 important feature and the feature we picked
combined_importance = dict(sorted_origin_importance[:5])


selected_feature_name = ["VAR_SDOHAlcoholUseDomainRisk_CAT",
                         "VAR_FinancialResourceStrainDomainCollected_FLG",
                         "VAR_FoodInsecurityDomainCollected_FLG",
                         "VAR_HousingStabilityDomainCollected_FLG",
                         "VAR_PhysicalActivityDomain_CAT",
                         "VAR_SafetyandDomesticViolenceDomainCollected_FLG",
                         "VAR_SDOHSocialConnectionDomain_CAT",
                         "VAR_StressDomain_CAT",
                         "VAR_TransportationDomainCollected_FLG",
                         "VAR_UtilitiesDomain_FLG"]
selected_importance = {}
for feature, importance in origin_importance.items():
    if feature in selected_feature_name:
        selected_importance[feature] = importance
print(selected_importance)
names = ["Alcohol Use",
         "Financial Resource",
         "Food Insecurity",
         "Housing Stability",
         "Physical Activity",
         "Safety",
         "Social Connection",
         "Stress",
         "Transportation",
         "Utilities"]
values = list(selected_importance.values())
new_selected_importance = dict(zip(names, values))
# print(new_selected_importance)

sorted_selected_importance = sorted(new_selected_importance.items(), key=lambda x: x[1], reverse=True)
combined_importance.update(dict(sorted_selected_importance))


features, scores = zip(*combined_importance.items())
plt.figure(figsize=(8, 6))
plt.barh(features, scores)
plt.xlabel('Feature Importance (Weight)', fontsize = 16)
plt.ylabel('Features', fontsize = 16)
plt.title('Variable Importance', fontsize = 16)
plt.gca().invert_yaxis()
plt.subplots_adjust(left=0.2, right=0.9)
plt.savefig("Importance_Reduced.pdf", format = "pdf")
plt.show()

# endregion

# region ===================== SHAP for XGBoost (Binary Classification) =====================
import shap

X_bg = shap.utils.sample(X_train, 200, random_state=42)

explainer = shap.Explainer(best_xgb, X_bg)
sv = explainer(X_test)

shap_values = sv.values
base_values = sv.base_values

if shap_values.ndim == 3:
    shap_values_pos = shap_values[:, :, 1]
    base_values_pos = base_values[:, 1] if np.ndim(base_values) == 2 else base_values
else:
    shap_values_pos = shap_values
    base_values_pos = base_values

# 1) Summary plot (beeswarm)
plt.figure()
shap.summary_plot(shap_values_pos, X_test, show=False, max_display=20)
plt.tight_layout()
plt.savefig("SHAP_Summary_Beeswarm.pdf", format="pdf")
plt.show()

# 2) Summary plot (bar)
plt.figure()
shap.summary_plot(shap_values_pos, X_test, plot_type="bar", show=False, max_display=20)
plt.tight_layout()
plt.savefig("SHAP_Summary_Bar.pdf", format="pdf")
plt.show()

# idx = 0
# # Waterfall
# plt.figure()
# shap.plots.waterfall(
#     shap.Explanation(
#         values=shap_values_pos[idx],
#         base_values=base_values_pos[idx] if np.ndim(base_values_pos) else base_values_pos,
#         data=X_test.iloc[idx],
#         feature_names=X_test.columns
#     ),
#     show=False
# )
# plt.tight_layout()
# plt.savefig("SHAP_Waterfall_OneCase.pdf", format="pdf")
# plt.show()

# Force plot
# shap.initjs()
# force = shap.force_plot(
#     base_values_pos[idx] if np.ndim(base_values_pos) else base_values_pos,
#     shap_values_pos[idx],
#     X_test.iloc[idx],
#     matplotlib=True
# )
# plt.savefig("SHAP_Force_OneCase.pdf", format="pdf")
# plt.show()

# 4) Dependence plot
mean_abs = np.mean(np.abs(shap_values_pos), axis=0)
top_feature = X_test.columns[int(np.argmax(mean_abs))]
plt.figure()
shap.dependence_plot(top_feature, shap_values_pos, X_test, show=False)
plt.tight_layout()
plt.savefig("SHAP_Dependence_TopFeature.pdf", format="pdf")
plt.show()

# endregion ===================== end SHAP =====================




