"""
preprocessing.py
-----------------
Merges application_record.csv with credit_record.csv, converts the monthly
payment-status history into a binary approve/decline label, cleans the
data, and prepares features for model training -- built around real credit
card underwriting factors: credit score, income, debt-to-income, employment.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

CATEGORICAL_COLS = ["EMPLOYMENT_STATUS", "GENDER", "EDUCATION_TYPE", "FAMILY_STATUS", "HOUSING_TYPE"]

NUMERIC_COLS = [
    "AGE", "ANNUAL_INCOME", "CREDIT_SCORE", "EMPLOYMENT_YEARS",
    "MONTHLY_DEBT", "DEBT_TO_INCOME", "EXISTING_ACCOUNTS",
]

FEATURE_COLS = NUMERIC_COLS + CATEGORICAL_COLS


def build_target_labels(credit_df: pd.DataFrame) -> pd.DataFrame:
    """STATUS 2-5 (60+ days past due) at any point -> high risk (TARGET=1)."""
    bad_status = {"2", "3", "4", "5"}
    credit_df = credit_df.copy()
    credit_df["IS_BAD_MONTH"] = credit_df["STATUS"].isin(bad_status).astype(int)

    target = (
        credit_df.groupby("ID")["IS_BAD_MONTH"]
        .max()
        .reset_index()
        .rename(columns={"IS_BAD_MONTH": "TARGET"})
    )
    return target


def load_and_merge(app_path="data/application_record.csv",
                    credit_path="data/credit_record.csv") -> pd.DataFrame:
    app_df = pd.read_csv(app_path)
    credit_df = pd.read_csv(credit_path, dtype={"STATUS": str})

    target_df = build_target_labels(credit_df)
    df = app_df.merge(target_df, on="ID", how="inner")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.drop_duplicates(subset="ID")

    df["ANNUAL_INCOME"] = df["ANNUAL_INCOME"].replace(0, np.nan)
    df["DEBT_TO_INCOME"] = (df["MONTHLY_DEBT"] * 12) / df["ANNUAL_INCOME"]
    df["DEBT_TO_INCOME"] = df["DEBT_TO_INCOME"].fillna(df["DEBT_TO_INCOME"].median())
    df["ANNUAL_INCOME"] = df["ANNUAL_INCOME"].fillna(df["ANNUAL_INCOME"].median())

    df = df.dropna(subset=NUMERIC_COLS + ["TARGET"], how="any")
    return df


def encode_features(df: pd.DataFrame, encoders: dict | None = None, fit=True):
    df = df.copy()
    encoders = encoders or {}

    for col in CATEGORICAL_COLS:
        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        else:
            le = encoders[col]
            df[col] = df[col].astype(str).map(
                lambda v: v if v in le.classes_ else le.classes_[0]
            )
            df[col] = le.transform(df[col])

    return df, encoders


def scale_features(X: pd.DataFrame, scaler: StandardScaler | None = None, fit=True):
    if fit:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)
    return X_scaled, scaler


def build_dataset(app_path="data/application_record.csv",
                   credit_path="data/credit_record.csv"):
    df = load_and_merge(app_path, credit_path)
    df = engineer_features(df)
    df_encoded, encoders = encode_features(df, fit=True)

    X = df_encoded[FEATURE_COLS]
    y = df_encoded["TARGET"]

    X_scaled, scaler = scale_features(X, fit=True)
    return X_scaled, y, encoders, scaler, df


if __name__ == "__main__":
    X_scaled, y, encoders, scaler, df = build_dataset()
    print(f"Clean dataset shape: {df.shape}")
    print(f"Feature matrix shape: {X_scaled.shape}")
    print(f"Target distribution:\n{y.value_counts(normalize=True)}")
