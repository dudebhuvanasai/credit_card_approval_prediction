# Ledger — Credit Card Approval Prediction

A small end-to-end app: a trained ML model behind a Flask API, with a
frontend where you fill out an applicant profile and get a live
approve/decline decision.

```
credit_approval_app/
├── data/                      # application_record.csv, credit_record.csv
├── notebook/
│   └── approval_prediction.ipynb   # rebuilt, leak-free training notebook
├── model/
│   ├── train_model.py         # same logic as the notebook, as a script
│   ├── credit_model.pkl       # trained model (generated)
│   ├── scaler.pkl             # StandardScaler for the linear model (generated)
│   ├── model_meta.pkl         # feature schema, threshold, metrics (generated)
│   └── metrics.json           # human-readable metrics (generated)
├── backend/
│   └── app.py                 # Flask API + serves the frontend
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
└── requirements.txt
```

## Run it

```bash
pip install -r requirements.txt

# 1. Train (writes model/*.pkl) — only needed once, or after changing the notebook
python model/train_model.py

# 2. Serve the API + frontend
python backend/app.py
```

Open `http://127.0.0.1:5000`.

## What changed vs. the original notebook

The original notebook merged applicant data with the **raw monthly**
credit-bureau table (one row per applicant per month) and predicted that
row's own status — so the label was effectively already inside the row
being scored, and the same applicant showed up in both train and test
under different months. Reported accuracy was ~99%, but the model had
learned nothing generalizable.

This version:
- Aggregates each applicant's credit history down to **one row per
  applicant** before any split happens.
- Only uses fields a bank actually has **at the moment someone applies**
  (income, job, family, housing, etc.) — no fields that require an
  account to already exist.
- Reports **Accuracy, Precision, Recall, F1, and ROC-AUC**, because ~98%
  of applicants are low-risk, so "approve everyone" already scores ~98%
  accuracy while catching 0% of risky applicants. Accuracy alone is not
  a meaningful metric here.
- Tunes the classification threshold instead of leaving it at the
  default 0.5, and picks the best model by ROC-AUC (more stable than F1
  with so few positive examples).

Final result: **Decision Tree**, ROC-AUC ≈ 0.68, ~97.6% accuracy at the
tuned threshold, with meaningfully-better-than-baseline precision/recall
on the minority "high risk" class. See `model/metrics.json` for the full
comparison across Logistic Regression, Decision Tree, and Random Forest.

Given how weak the raw application-time signal is for this dataset (this
is a known property of this particular Kaggle dataset, not a bug), these
numbers are the honest ceiling for a leak-free model — treat this as a
demo of the full pipeline rather than a production underwriting model.

## API

`POST /api/predict`

```json
{
  "gender": "F", "own_car": "N", "own_realty": "Y",
  "children": 0, "family_members": 2, "family_status": "Married",
  "annual_income": 48000, "income_type": "Working",
  "occupation": "Core staff", "is_employed": true, "years_employed": 6,
  "age": 34, "education": "Higher education",
  "housing_type": "House / apartment",
  "has_work_phone": false, "has_phone": true, "has_email": false
}
```

Response:

```json
{
  "decision": "APPROVED",
  "risk_probability": 0.185,
  "decision_threshold": 0.95,
  "model_used": "decision_tree"
}
```
