"""
generate_dataset.py
--------------------
Generates a synthetic Indian credit card applicant dataset built around the
factors real Indian card issuers underwrite on: annual income (INR), CIBIL
score (300-900 scale), existing debt, employment type/length, and number of
open credit accounts.

Two files are produced, matching the two-table structure real bureau data
comes in:
  - application_record.csv : one row per applicant (their profile)
  - credit_record.csv      : monthly payment-history rows per applicant,
                              used to derive the approve/decline label

Run: python generate_dataset.py
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N_APPLICANTS = 6000

EMPLOYMENT_STATUSES = ["Salaried", "Self-employed", "Retired", "Unemployed", "Student"]
GENDERS = ["M", "F"]
EDUCATION_TYPES = ["Below High School", "High School", "Graduate", "Post Graduate", "Professional Degree"]
FAMILY_STATUSES = ["Married", "Single", "Divorced", "Widowed"]
HOUSING_TYPES = ["Owned", "Rented", "Living with Parents", "Company Provided"]


def generate_application_record(n=N_APPLICANTS):
    ids = np.arange(100000, 100000 + n)

    age = RNG.integers(18, 75, size=n)
    gender = RNG.choice(GENDERS, size=n, p=[0.54, 0.46])
    family_status = RNG.choice(FAMILY_STATUSES, size=n, p=[0.60, 0.28, 0.08, 0.04])
    housing_type = RNG.choice(HOUSING_TYPES, size=n, p=[0.40, 0.32, 0.20, 0.08])

    education_type = RNG.choice(
        EDUCATION_TYPES, size=n, p=[0.06, 0.32, 0.38, 0.18, 0.06]
    )
    edu_score_bump = pd.Series(education_type).map({
        "Below High School": -45, "High School": -10, "Graduate": 15,
        "Post Graduate": 35, "Professional Degree": 50,
    }).to_numpy()
    edu_income_bump = pd.Series(education_type).map({
        "Below High School": -120000, "High School": -40000, "Graduate": 0,
        "Post Graduate": 90000, "Professional Degree": 180000,
    }).to_numpy()

    employment_status = RNG.choice(
        EMPLOYMENT_STATUSES, size=n, p=[0.60, 0.18, 0.14, 0.06, 0.02]
    )

    employment_years = np.clip(RNG.normal((age - 18) * 0.3, 3), 0, None)
    employment_years = np.where(employment_status == "Unemployed", 0, employment_years)
    employment_years = np.where(employment_status == "Student", np.clip(employment_years, 0, 2), employment_years)
    employment_years = employment_years.round(1)

    # Annual income in INR. Salaried median roughly ~6 LPA, self-employed
    # more variance (ITR-reported), retired lower (pension), students/
    # unemployed near zero independent income. Education level shifts income
    # up/down the same way it does in real hiring/pay data.
    base_income = RNG.normal(500000, 220000, size=n)
    status_bump = pd.Series(employment_status).map({
        "Salaried": 0, "Self-employed": 90000, "Retired": -280000,
        "Unemployed": -480000, "Student": -470000,
    }).to_numpy()
    annual_income = np.clip(
        base_income + status_bump + edu_income_bump + employment_years * 9000, 0, None
    ).round(-2)
    # Students/unemployed: mostly no independent income, occasional small stipend
    annual_income = np.where(
        employment_status == "Unemployed", RNG.choice([0, 0, 0, 20000, 40000], size=n), annual_income
    )
    annual_income = np.where(
        employment_status == "Student", RNG.choice([0, 0, 30000, 60000, 90000], size=n), annual_income
    )

    # CIBIL score: 300-900 scale, 750+ considered good/eligible by most banks
    score_base = RNG.normal(700, 85, size=n)
    score_income_bump = (np.clip(annual_income, 0, 2000000) - 500000) / 25000
    score_employ_bump = employment_years * 2.5
    unemployed_penalty = np.where(employment_status.astype(str) == "Unemployed", -70, 0)
    credit_score = np.clip(
        score_base + score_income_bump + score_employ_bump + edu_score_bump + unemployed_penalty,
        300, 900,
    ).round(0).astype(int)
    # Students/unemployed with no credit history often have no/thin CIBIL file
    credit_score = np.where(annual_income <= 0, np.clip(credit_score, 300, 650), credit_score)

    monthly_debt = np.clip(
        RNG.normal(annual_income / 12 * 0.16, annual_income / 12 * 0.08), 0, None
    ).round(-2)

    existing_accounts = RNG.integers(0, 8, size=n)
    requested_credit_limit = RNG.choice(
        [25000, 50000, 75000, 100000, 150000, 200000, 300000, 500000], size=n
    )

    df = pd.DataFrame({
        "ID": ids,
        "AGE": age,
        "GENDER": gender,
        "ANNUAL_INCOME": annual_income,
        "CREDIT_SCORE": credit_score,
        "EMPLOYMENT_STATUS": employment_status,
        "EMPLOYMENT_YEARS": employment_years,
        "EDUCATION_TYPE": education_type,
        "FAMILY_STATUS": family_status,
        "HOUSING_TYPE": housing_type,
        "MONTHLY_DEBT": monthly_debt,
        "EXISTING_ACCOUNTS": existing_accounts,
        "REQUESTED_CREDIT_LIMIT": requested_credit_limit,
    })
    return df, credit_score, annual_income, monthly_debt


def generate_credit_record(app_df, credit_score, annual_income, monthly_debt):
    """
    Monthly payment-history rows, with STATUS codes matching real bureau data:
      0-5 -> days-past-due buckets (2-5 = 60+ days late / high risk)
      C   -> paid off that month     X -> no loan activity that month
    Risk is driven by CIBIL score and debt-to-income ratio -- the same two
    numbers Indian issuers weight most heavily.
    """
    rows = []
    dti = (monthly_debt * 12) / np.clip(annual_income, 1, None)

    score_z = (credit_score - credit_score.mean()) / credit_score.std()
    dti_z = (dti - dti.mean()) / dti.std()
    risk_score = -1.6 * score_z + 1.3 * dti_z

    # Quantile-based thresholding gives direct control over the resulting
    # approval rate (top ~22% riskiest applicants become "risky"), rather
    # than relying on the logistic curve's shape, which can drift the
    # overall approve/decline split further than intended.
    threshold = np.quantile(risk_score, 0.78)
    is_risky = risk_score > threshold

    for i, applicant_id in enumerate(app_df["ID"]):
        n_months = RNG.integers(6, 25)
        p_bad_month = 0.22 if is_risky[i] else 0.015
        for m in range(n_months):
            months_balance = -m
            roll = RNG.random()
            if roll < p_bad_month:
                status = str(RNG.integers(2, 6))
            elif roll < p_bad_month + 0.15:
                status = RNG.choice(["0", "1"])
            else:
                status = RNG.choice(["C", "X"], p=[0.75, 0.25])
            rows.append((applicant_id, months_balance, status))

    return pd.DataFrame(rows, columns=["ID", "MONTHS_BALANCE", "STATUS"])


def main():
    app_df, credit_score, annual_income, monthly_debt = generate_application_record()
    credit_df = generate_credit_record(app_df, credit_score, annual_income, monthly_debt)

    app_df.to_csv("data/application_record.csv", index=False)
    credit_df.to_csv("data/credit_record.csv", index=False)
    print(f"Wrote data/application_record.csv  ({len(app_df):,} applicants)")
    print(f"Wrote data/credit_record.csv       ({len(credit_df):,} monthly records)")


if __name__ == "__main__":
    main()
