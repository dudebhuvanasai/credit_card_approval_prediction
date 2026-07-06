"""
Credit Card Approval Prediction - Backend API
==============================================
Flask app that:
  1. Serves the frontend (static HTML/CSS/JS).
  2. Exposes POST /api/predict which takes an applicant's details and
     returns an approval decision + risk probability, using the model
     trained in model/train_model.py.

Run:
    pip install -r requirements.txt
    python backend/app.py
Then open http://127.0.0.1:5000
"""

import os

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "model")
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")

# ---------------------------------------------------------------------------
# Load trained artifacts once at startup
# ---------------------------------------------------------------------------
model = joblib.load(os.path.join(MODEL_DIR, "credit_model.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
meta = joblib.load(os.path.join(MODEL_DIR, "model_meta.pkl"))

FEATURE_COLUMNS = meta["feature_columns"]
CATEGORICAL_COLS = meta["categorical_cols"]
USES_SCALED_INPUT = meta["uses_scaled_input"]
DECISION_THRESHOLD = meta["decision_threshold"]
BEST_MODEL_NAME = meta["best_model_name"]

REQUIRED_FIELDS = [
    "gender",            # "M" or "F"
    "own_car",           # "Y" or "N"
    "own_realty",        # "Y" or "N"
    "children",          # int
    "annual_income",     # float
    "income_type",       # str, one of the trained categories
    "education",         # str
    "family_status",     # str
    "housing_type",      # str
    "age",                # int (years)
    "years_employed",    # float, 0 if not currently employed
    "is_employed",       # bool
    "family_members",    # int
    "occupation",        # str
    "has_work_phone",    # bool
    "has_phone",         # bool
    "has_email",         # bool
]


def build_feature_row(payload: dict) -> pd.DataFrame:
    """Convert raw applicant JSON into the exact one-hot column layout the
    model was trained on."""

    income = float(payload["annual_income"])
    family_members = max(int(payload["family_members"]), 1)

    raw = {
        "CNT_CHILDREN": int(payload["children"]),
        "AMT_INCOME_TOTAL": income,
        "FLAG_WORK_PHONE": int(bool(payload["has_work_phone"])),
        "FLAG_PHONE": int(bool(payload["has_phone"])),
        "FLAG_EMAIL": int(bool(payload["has_email"])),
        "CNT_FAM_MEMBERS": family_members,
        "AGE_YEARS": float(payload["age"]),
        "IS_EMPLOYED": int(bool(payload["is_employed"])),
        "YEARS_EMPLOYED": float(payload["years_employed"]) if payload["is_employed"] else 0.0,
        "INCOME_PER_FAMILY_MEMBER": round(income / family_members, 2),
        "CODE_GENDER": payload["gender"],
        "FLAG_OWN_CAR": payload["own_car"],
        "FLAG_OWN_REALTY": payload["own_realty"],
        "NAME_INCOME_TYPE": payload["income_type"],
        "NAME_EDUCATION_TYPE": payload["education"],
        "NAME_FAMILY_STATUS": payload["family_status"],
        "NAME_HOUSING_TYPE": payload["housing_type"],
        "OCCUPATION_TYPE": payload.get("occupation") or "Unknown",
    }

    row_df = pd.DataFrame([raw])
    row_encoded = pd.get_dummies(row_df, columns=CATEGORICAL_COLS)

    # Align to the exact training-time column layout (missing dummy columns
    # become 0, unseen categories are simply dropped).
    row_aligned = row_encoded.reindex(columns=FEATURE_COLUMNS, fill_value=0)
    return row_aligned


@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/api/health")
def health():
    acc = meta["metrics"].get(BEST_MODEL_NAME, {}).get("accuracy")
    return jsonify(
        {
            "status": "ok",
            "model": BEST_MODEL_NAME,
            "decision_threshold": DECISION_THRESHOLD,
            "accuracy": acc,
        }
    )


@app.route("/api/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True) or {}

    missing = [f for f in REQUIRED_FIELDS if f not in payload]
    if missing:
        return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400

    try:
        X_row = build_feature_row(payload)
    except (ValueError, TypeError, KeyError) as exc:
        return jsonify({"error": f"Invalid input: {exc}"}), 400

    X_input = scaler.transform(X_row) if USES_SCALED_INPUT else X_row

    risk_probability = float(model.predict_proba(X_input)[0, 1])
    is_high_risk = risk_probability >= DECISION_THRESHOLD

    decision = "REJECTED" if is_high_risk else "APPROVED"

    return jsonify(
        {
            "decision": decision,
            "risk_probability": round(risk_probability, 4),
            "decision_threshold": DECISION_THRESHOLD,
            "model_used": BEST_MODEL_NAME,
        }
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
