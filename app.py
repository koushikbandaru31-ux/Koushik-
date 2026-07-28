"""
app.py
------
Flask web application for the Credit Card Approval Prediction model.

Routes:
  /            Home page
  /register    Create an account
  /login       Sign in
  /logout      Sign out
  /predict     Application form (GET) / prediction result (POST)
  /history     Past predictions for the logged-in user

Run: python app.py
Then open http://127.0.0.1:5000
"""

import json
import os
import pickle

import numpy as np
from flask import Flask, redirect, render_template, request, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash

import db
from preprocessing import CATEGORICAL_COLS, FEATURE_COLS

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-key-change-in-production")

db.init_db()
db.seed_ml_models()
db.seed_demo_user()

with open("model/best_model.pkl", "rb") as f:
    MODEL = pickle.load(f)
with open("model/encoders.pkl", "rb") as f:
    ENCODERS = pickle.load(f)
with open("model/scaler.pkl", "rb") as f:
    SCALER = pickle.load(f)
with open("model/model_info.json") as f:
    MODEL_INFO = json.load(f)

OPTIONS = {col: list(ENCODERS[col].classes_) for col in CATEGORICAL_COLS}

# --- Hard eligibility rules (checked before any ML scoring) -----------------
# Real Indian card issuers apply baseline policy rules before risk-scoring an
# applicant at all. A statistical model has no built-in notion of these --
# it will happily "approve" a 10-year-old if the other numbers look fine, so
# these checks come first and can decline an applicant outright.
#
# Rules below reflect common Indian bank policy (these vary slightly by
# issuer -- e.g. HDFC, SBI Card, ICICI, Axis -- but the pattern is standard):
MIN_AGE = 18                          # no card can be issued to a minor
MIN_AGE_FOR_UNSECURED_CARD = 21       # under 21 usually needs a secured/add-on card instead
MAX_AGE_SALARIED = 60                 # typical upper age cap, salaried applicants
MAX_AGE_SELF_EMPLOYED = 65            # typical upper age cap, self-employed applicants
MIN_CIBIL_SCORE = 650                 # most issuers reject outright below this (750+ preferred)

MIN_INCOME_BY_STATUS = {              # typical minimum annual income (INR) by employment type
    "Salaried": 180000,               # ~₹15,000/month
    "Self-employed": 240000,          # ITR-proven income, ~₹20,000/month
    "Retired": 180000,                # pension income
}


def check_hard_eligibility(age, annual_income, employment_status, credit_score):
    """Returns a list of hard-decline reasons. Empty list = passes the gate."""
    reasons = []

    if age < MIN_AGE:
        reasons.append(f"RBI regulations do not permit a credit card to be issued to "
                        f"anyone under {MIN_AGE} years of age.")
        return reasons  # no point checking anything else

    if employment_status in ("Student", "Unemployed") and annual_income <= 0:
        reasons.append(
            "Applicants with no independent income are not eligible for a regular "
            "unsecured credit card. A secured card against a Fixed Deposit is the "
            "usual alternative for students and those without independent income."
        )

    elif age < MIN_AGE_FOR_UNSECURED_CARD and annual_income < MIN_INCOME_BY_STATUS.get("Salaried", 180000):
        reasons.append(
            f"Applicants under {MIN_AGE_FOR_UNSECURED_CARD} without proof of independent "
            f"income are usually not issued a primary unsecured card. A secured card "
            f"against a Fixed Deposit, or an add-on card linked to a parent/guardian's "
            f"account, is the typical route instead."
        )

    if employment_status == "Salaried" and age > MAX_AGE_SALARIED:
        reasons.append(
            f"Most Indian issuers cap primary credit card issuance for salaried "
            f"applicants at {MAX_AGE_SALARIED} years."
        )
    elif employment_status == "Self-employed" and age > MAX_AGE_SELF_EMPLOYED:
        reasons.append(
            f"Most Indian issuers cap primary credit card issuance for self-employed "
            f"applicants at {MAX_AGE_SELF_EMPLOYED} years."
        )

    min_income = MIN_INCOME_BY_STATUS.get(employment_status)
    if min_income and annual_income < min_income:
        reasons.append(
            f"Most issuers require a minimum annual income of ₹{min_income:,} for "
            f"{employment_status.lower()} applicants; this profile falls short of that."
        )

    if credit_score < MIN_CIBIL_SCORE:
        reasons.append(
            f"Most Indian card issuers require a CIBIL score of at least "
            f"{MIN_CIBIL_SCORE} (750+ is preferred) for an unsecured card. A score "
            f"below this is typically declined outright regardless of income."
        )

    return reasons


def explain_decision(row, approved):
    """
    Builds a short, human-readable list of reasons behind the model's
    approve/decline decision, based on where the applicant's key numbers
    sit relative to typical underwriting thresholds.
    """
    dti_pct = row["DEBT_TO_INCOME"] * 100
    score = row["CREDIT_SCORE"]
    emp_years = row["EMPLOYMENT_YEARS"]
    accounts = row["EXISTING_ACCOUNTS"]

    positives, negatives = [], []

    if score >= 750:
        positives.append(f"CIBIL score of {int(score)} is in the 'excellent' range (750+).")
    elif score >= 700:
        positives.append(f"CIBIL score of {int(score)} is comfortably above the 650 minimum.")
    else:
        negatives.append(f"CIBIL score of {int(score)} is close to the minimum most issuers accept.")

    if dti_pct <= 20:
        positives.append(f"Debt-to-income ratio is low ({dti_pct:.0f}%), leaving healthy repayment capacity.")
    elif dti_pct <= 40:
        negatives.append(f"Debt-to-income ratio is moderate ({dti_pct:.0f}%).")
    else:
        negatives.append(f"Debt-to-income ratio is high ({dti_pct:.0f}%), which raises repayment risk.")

    if emp_years >= 3:
        positives.append(f"{emp_years:.1f} years in current employment shows income stability.")
    elif emp_years < 1:
        negatives.append(f"Only {emp_years:.1f} years in current employment is a limited track record.")

    if accounts >= 6:
        negatives.append(f"{int(accounts)} existing credit accounts add to overall credit exposure.")
    elif accounts <= 1:
        positives.append(f"Only {int(accounts)} existing credit account(s), indicating low exposure.")

    reasons = positives if approved else negatives
    if not reasons:
        reasons = positives + negatives
    return reasons[:3] or ["Based on the overall profile relative to similar applicants in the model's training data."]


def current_user():
    user_id = session.get("user_id")
    if not user_id:
        return None
    return db.get_user_by_id(user_id)


@app.context_processor
def inject_user():
    return {"current_user": current_user()}


@app.route("/")
def home():
    return render_template("index.html", model_info=MODEL_INFO)


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "GET":
        return render_template("register.html")

    name = request.form.get("name", "").strip()
    email = request.form.get("email", "").strip().lower()
    password = request.form.get("password", "")

    if not name or not email or len(password) < 6:
        return render_template("register.html",
                                error="Please fill in all fields; password must be at least 6 characters.")

    if db.get_user_by_email(email):
        return render_template("register.html", error="An account with that email already exists.")

    user_id = db.create_user(name, email, generate_password_hash(password))
    session["user_id"] = user_id
    return redirect(url_for("home"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "GET":
        return render_template("login.html")

    email = request.form.get("email", "").strip().lower()
    password = request.form.get("password", "")

    user = db.get_user_by_email(email)
    if not user or not check_password_hash(user["Password"], password):
        return render_template("login.html", error="Incorrect email or password.")

    session["user_id"] = user["UserID"]
    return redirect(url_for("home"))


@app.route("/logout")
def logout():
    session.pop("user_id", None)
    return redirect(url_for("home"))


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return render_template("predict.html", options=OPTIONS, form=None)

    form = request.form.to_dict()

    try:
        annual_income = float(form["annual_income"])
        monthly_debt = float(form["monthly_debt"])
        row = {
            "AGE": float(form["age"]),
            "GENDER": form["gender"],
            "ANNUAL_INCOME": annual_income,
            "CREDIT_SCORE": float(form["credit_score"]),
            "EMPLOYMENT_YEARS": float(form["employment_years"]),
            "MONTHLY_DEBT": monthly_debt,
            "DEBT_TO_INCOME": (monthly_debt * 12) / max(annual_income, 1),
            "EXISTING_ACCOUNTS": float(form["existing_accounts"]),
            "EMPLOYMENT_STATUS": form["employment_status"],
            "EDUCATION_TYPE": form["education_type"],
            "FAMILY_STATUS": form["family_status"],
            "HOUSING_TYPE": form["housing_type"],
        }
    except (KeyError, ValueError) as e:
        return render_template("predict.html", options=OPTIONS, form=form,
                                error=f"Invalid input: {e}")

    # Hard eligibility gate -- runs before any ML risk scoring
    decline_reasons = check_hard_eligibility(
        age=row["AGE"], annual_income=row["ANNUAL_INCOME"],
        employment_status=row["EMPLOYMENT_STATUS"], credit_score=row["CREDIT_SCORE"],
    )
    if decline_reasons:
        result = {
            "approved": False,
            "hard_decline": True,
            "risk_score": 100.0,
            "reasons": decline_reasons,
        }
        if session.get("user_id"):
            db.save_applicant_and_prediction(
                session["user_id"], row, MODEL_INFO["best_model_name"],
                "Rejected", 100.0, decline_reasons,
            )
        return render_template("predict.html", options=OPTIONS, form=form, result=result)

    values = []
    for col in FEATURE_COLS:
        val = row[col]
        if col in CATEGORICAL_COLS:
            le = ENCODERS[col]
            if val not in le.classes_:
                val = le.classes_[0]
            val = le.transform([val])[0]
        values.append(float(val))

    X = np.array(values).reshape(1, -1)
    X_scaled = SCALER.transform(X)

    proba_reject = MODEL.predict_proba(X_scaled)[0, 1]
    prediction = int(proba_reject >= 0.5)
    approved = prediction == 0

    reasons = explain_decision(row, approved)
    result = {
        "approved": approved,
        "hard_decline": False,
        "confidence": round((1 - proba_reject) * 100 if approved else proba_reject * 100, 1),
        "risk_score": round(proba_reject * 100, 1),
        "reasons": reasons,
    }

    if session.get("user_id"):
        db.save_applicant_and_prediction(
            session["user_id"], row, MODEL_INFO["best_model_name"],
            "Approved" if approved else "Rejected", result["risk_score"], reasons,
        )

    return render_template("predict.html", options=OPTIONS, form=form, result=result)


@app.route("/history")
def history():
    user = current_user()
    if not user:
        return redirect(url_for("login"))

    rows = db.get_prediction_history(user["UserID"])
    records = []
    for r in rows:
        records.append({
            "date": r["PredictionDate"][:16].replace("T", " "),
            "result": r["ApprovalResult"],
            "risk_score": r["RiskScore"],
            "reasons": json.loads(r["Reasons"]) if r["Reasons"] else [],
            "age": r["Age"], "gender": r["Gender"], "income": r["AnnualIncome"],
            "credit_score": r["CreditScore"], "employment_status": r["EmploymentStatus"],
            "model_name": r["ModelName"],
        })
    return render_template("history.html", records=records)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("FLASK_DEBUG", "true").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug_mode)
