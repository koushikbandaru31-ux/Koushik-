"""
eda.py
------
Generates count plots and distribution plots exploring relationships
between applicant profiles and credit card approval outcomes.
Saves figures into notebooks/figures/.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from preprocessing import engineer_features, load_and_merge

sns.set_theme(style="whitegrid")
OUT_DIR = "notebooks/figures"
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    df = load_and_merge()
    df = engineer_features(df)
    df["APPROVAL"] = df["TARGET"].map({0: "Approved", 1: "Declined"})

    plt.figure(figsize=(5, 4))
    sns.countplot(data=df, x="APPROVAL", hue="APPROVAL", palette="Set2", legend=False)
    plt.title("Approval vs Decline Counts")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/target_balance.png", dpi=110)
    plt.close()

    plt.figure(figsize=(7, 4))
    sns.countplot(data=df, y="EMPLOYMENT_STATUS", hue="APPROVAL", palette="Set2")
    plt.title("Approval by Employment Status")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/approval_by_employment.png", dpi=110)
    plt.close()

    plt.figure(figsize=(6, 4))
    sns.histplot(data=df, x="CREDIT_SCORE", hue="APPROVAL", bins=40,
                 kde=True, palette="Set2", element="step")
    plt.title("Credit Score Distribution by Approval Outcome")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/credit_score_distribution.png", dpi=110)
    plt.close()

    plt.figure(figsize=(6, 4))
    sns.histplot(data=df, x="ANNUAL_INCOME", hue="APPROVAL", bins=40,
                 kde=True, palette="Set2", element="step")
    plt.xlim(0, df["ANNUAL_INCOME"].quantile(0.98))
    plt.title("Income Distribution by Approval Outcome")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/income_distribution.png", dpi=110)
    plt.close()

    plt.figure(figsize=(6, 4))
    sns.histplot(data=df, x="DEBT_TO_INCOME", hue="APPROVAL", bins=40,
                 kde=True, palette="Set2", element="step")
    plt.xlim(0, df["DEBT_TO_INCOME"].quantile(0.98))
    plt.title("Debt-to-Income Ratio by Approval Outcome")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/dti_distribution.png", dpi=110)
    plt.close()

    plt.figure(figsize=(6, 4))
    sns.histplot(data=df, x="EMPLOYMENT_YEARS", hue="APPROVAL", bins=30,
                 kde=True, palette="Set2", element="step")
    plt.title("Employment Length by Approval Outcome")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/employment_distribution.png", dpi=110)
    plt.close()

    plt.figure(figsize=(7, 4))
    sns.countplot(data=df, y="EDUCATION_TYPE", hue="APPROVAL", palette="Set2")
    plt.title("Approval by Education Level")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/approval_by_education.png", dpi=110)
    plt.close()

    print(f"Saved 7 figures to {OUT_DIR}/")


if __name__ == "__main__":
    main()
