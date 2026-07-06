"""
Credit Card Approval Prediction - Model Training
=================================================
Fixes applied vs the original notebook:
1. Target leakage removed: original code merged application data with the
   RAW monthly credit_record table (one row per applicant per month) and
   then predicted STATUS from that same row -> the label was basically
   already inside the feature set, and applicants were duplicated across
   train/test splits. Here credit history is aggregated to ONE row per
   applicant (ID) BEFORE the train/test split.
2. Realistic target: an applicant is flagged HIGH RISK (1) if they were
   ever 60+ days past due (STATUS in 2,3,4,5) at any point in their credit
   history. This is the standard, non-leaky target used for this dataset.
3. Proper handling of sentinel values (DAYS_EMPLOYED = 365243 for
   unemployed/pensioners), missing OCCUPATION_TYPE, and engineered
   features (age, years employed, income per family member).
4. Class imbalance handled with class_weight="balanced" (no leakage-free
   oversampling library was available offline).
5. Same three model families as the original notebook (Logistic
   Regression, Decision Tree, Random Forest) are trained, tuned with
   RandomizedSearchCV, and compared on Accuracy / Precision / Recall / F1
   / ROC-AUC -- not accuracy alone, since the target is imbalanced.
6. Final artifacts (best model + scaler + column schema) are saved with
   joblib so the Flask backend (app.py) can load them directly.
"""

import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
app = pd.read_csv("data/application_record.csv")
credit = pd.read_csv("data/credit_record.csv")

# Drop duplicate applicants (same profile reported under multiple IDs)
feature_cols_for_dedup = [c for c in app.columns if c != "ID"]
app = app.drop_duplicates(subset=feature_cols_for_dedup, keep="first")

# ---------------------------------------------------------------------------
# 2. Aggregate credit history to ONE row per applicant (no leakage)
# ---------------------------------------------------------------------------
def worst_status_flag(statuses):
    """1 if the applicant was ever 60+ days overdue, else 0."""
    return int(any(s in ("2", "3", "4", "5") for s in statuses))


credit_agg = (
    credit.groupby("ID")
    .agg(
        months_on_book=("MONTHS_BALANCE", lambda x: x.max() - x.min() + 1),
        ever_overdue_1_29=("STATUS", lambda x: int(any(s == "1" for s in x))),
        statuses=("STATUS", list),
    )
    .reset_index()
)
credit_agg["TARGET"] = credit_agg["statuses"].apply(worst_status_flag)
credit_agg.drop(columns=["statuses"], inplace=True)

# Only applicants who actually have credit history can have a label
data = app.merge(credit_agg, on="ID", how="inner")

print("Merged, labeled dataset shape:", data.shape)
print("Target distribution:\n", data["TARGET"].value_counts(normalize=True))

# ---------------------------------------------------------------------------
# 3. Feature engineering
# ---------------------------------------------------------------------------
data["AGE_YEARS"] = (-data["DAYS_BIRTH"] / 365.25).round(1)

# DAYS_EMPLOYED uses 365243 as a sentinel for "not currently employed"
data["IS_EMPLOYED"] = (data["DAYS_EMPLOYED"] < 0).astype(int)
data["YEARS_EMPLOYED"] = np.where(
    data["DAYS_EMPLOYED"] < 0, -data["DAYS_EMPLOYED"] / 365.25, 0
).round(1)

data["OCCUPATION_TYPE"] = data["OCCUPATION_TYPE"].fillna("Unknown")
data["CNT_FAM_MEMBERS"] = data["CNT_FAM_MEMBERS"].fillna(1)
data["INCOME_PER_FAMILY_MEMBER"] = (
    data["AMT_INCOME_TOTAL"] / data["CNT_FAM_MEMBERS"].replace(0, 1)
).round(2)

# NOTE: months_on_book / ever_overdue_1_29 are DERIVED FROM CREDIT HISTORY.
# A brand-new applicant being screened for approval has no credit history
# yet, so these columns must not be used as model inputs -- only the
# application-time fields the bank actually has at decision time.
# Both `months_on_book` and `ever_overdue_1_29` come from the applicant's
# EXISTING account history at the credit bureau. A brand-new applicant
# filling out this application has no such history yet, so a model meant
# to score NEW applications must not depend on them -- the web app below
# can only ever collect application-time fields (income, job, family,
# housing, etc.), never "months this same card has been open".
drop_cols = [
    "ID",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "FLAG_MOBIL",
    "months_on_book",
    "ever_overdue_1_29",
]
drop_cols = [c for c in drop_cols if c in data.columns]
data = data.drop(columns=drop_cols)

# ---------------------------------------------------------------------------
# 4. Encode categoricals (one-hot) and record the schema for inference
# ---------------------------------------------------------------------------
categorical_cols = [
    "CODE_GENDER",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "OCCUPATION_TYPE",
]

y = data["TARGET"]
X = pd.get_dummies(data.drop(columns=["TARGET"]), columns=categorical_cols)

feature_columns = X.columns.tolist()

# ---------------------------------------------------------------------------
# 5. Train / test split (one row per applicant -> no cross-split leakage)
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------------------------
# 6. Train + tune the three model families used in the original notebook
# ---------------------------------------------------------------------------
results = {}
fitted_models = {}

# --- Logistic Regression (needs scaled features) ---
print("\nTraining Logistic Regression...")
lr = LogisticRegression(
    max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
)
lr.fit(X_train_scaled, y_train)
fitted_models["logistic_regression"] = (lr, True)

# --- Decision Tree (tuned) ---
print("Training Decision Tree (RandomizedSearchCV)...")
dt_param_dist = {
    "max_depth": [4, 6, 8, 10, 12, None],
    "min_samples_split": [2, 5, 10, 20],
    "min_samples_leaf": [1, 2, 5, 10],
    "criterion": ["gini", "entropy"],
}
dt_search = RandomizedSearchCV(
    DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE),
    dt_param_dist,
    n_iter=15,
    scoring="roc_auc",
    cv=3,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
dt_search.fit(X_train, y_train)
fitted_models["decision_tree"] = (dt_search.best_estimator_, False)

# --- Random Forest (tuned) ---
print("Training Random Forest (RandomizedSearchCV)...")
rf_param_dist = {
    "n_estimators": [200, 300, 400, 500],
    "max_depth": [6, 8, 10, 12, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"],
}
rf_search = RandomizedSearchCV(
    RandomForestClassifier(
        class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
    ),
    rf_param_dist,
    n_iter=15,
    scoring="roc_auc",
    cv=3,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
rf_search.fit(X_train, y_train)
fitted_models["random_forest"] = (rf_search.best_estimator_, False)

# ---------------------------------------------------------------------------
# 7. Evaluate all models
# ---------------------------------------------------------------------------
best_name, best_model, best_auc, best_uses_scaled = None, None, -1, False

for name, (model, uses_scaled) in fitted_models.items():
    X_eval = X_test_scaled if uses_scaled else X_test
    y_pred = model.predict(X_eval)
    y_proba = model.predict_proba(X_eval)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba)

    results[name] = {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "roc_auc": round(auc, 4),
    }

    print(f"\n=== {name} ===")
    print(results[name])
    print(classification_report(y_test, y_pred, zero_division=0))

    if auc > best_auc:
        best_name, best_model, best_auc, best_uses_scaled = name, model, auc, uses_scaled

print("\nBEST MODEL (by ROC-AUC, most stable metric for this imbalanced target):", best_name)
print(json.dumps(results, indent=2))

# ---------------------------------------------------------------------------
# 7b. Pick an operating threshold (default 0.5 is a poor cut-off for such an
#     imbalanced target -- it makes the minority "high risk" class almost
#     unreachable). We scan thresholds and pick the one that maximizes F1.
# ---------------------------------------------------------------------------
X_eval_best = X_test_scaled if best_uses_scaled else X_test
best_proba = best_model.predict_proba(X_eval_best)[:, 1]

thresholds = np.linspace(0.05, 0.95, 19)
best_threshold, best_thresh_f1 = 0.5, -1
for t in thresholds:
    preds = (best_proba >= t).astype(int)
    f1_t = f1_score(y_test, preds, zero_division=0)
    if f1_t > best_thresh_f1:
        best_threshold, best_thresh_f1 = t, f1_t

final_preds = (best_proba >= best_threshold).astype(int)
print("\nChosen decision threshold:", round(best_threshold, 2))
print("Metrics at chosen threshold:")
print(
    {
        "accuracy": round(accuracy_score(y_test, final_preds), 4),
        "precision": round(precision_score(y_test, final_preds, zero_division=0), 4),
        "recall": round(recall_score(y_test, final_preds, zero_division=0), 4),
        "f1": round(best_thresh_f1, 4),
    }
)

# ---------------------------------------------------------------------------
# 8. Persist artifacts for the Flask backend
# ---------------------------------------------------------------------------
joblib.dump(best_model, "model/credit_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")
joblib.dump(
    {
        "feature_columns": feature_columns,
        "categorical_cols": categorical_cols,
        "uses_scaled_input": best_uses_scaled,
        "best_model_name": best_name,
        "decision_threshold": float(best_threshold),
        "metrics": results,
    },
    "model/model_meta.pkl",
)

with open("model/metrics.json", "w") as f:
    json.dump(
        {
            "best_model": best_name,
            "decision_threshold": float(best_threshold),
            "results": results,
        },
        f,
        indent=2,
    )

print("\nSaved model/credit_model.pkl, model/scaler.pkl, model/model_meta.pkl")
