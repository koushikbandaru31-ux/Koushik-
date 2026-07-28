"""
db.py
-----
SQLite database layer implementing the project's ER diagram:

  Users (1) ----< Applicant_Details (1) ----< Credit_History
                        |
                        v (1:1)
                 Approval_Prediction >---- (N:1) ML_Model

Tables:
  users               UserID, Name, Email, Password (hashed), Role
  applicant_details   ApplicantID, UserID (FK), applicant profile fields
  credit_history      HistoryID, ApplicantID (FK) -- reference table describing
                       the monthly payment-history data used to train the
                       model (not populated per web-request; see README)
  ml_model            ModelID, ModelName, AlgorithmType, Accuracy, ModelFile
  approval_prediction PredictionID, ApplicantID (FK), ModelID (FK),
                       ApprovalResult, RiskScore, PredictionDate
"""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime

DB_PATH = "database.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    UserID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT NOT NULL,
    Email TEXT UNIQUE NOT NULL,
    Password TEXT NOT NULL,
    Role TEXT NOT NULL DEFAULT 'analyst',
    CreatedAt TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS applicant_details (
    ApplicantID INTEGER PRIMARY KEY AUTOINCREMENT,
    UserID INTEGER,
    Age INTEGER,
    Gender TEXT,
    AnnualIncome REAL,
    CreditScore INTEGER,
    EmploymentStatus TEXT,
    EmploymentYears REAL,
    EducationType TEXT,
    FamilyStatus TEXT,
    HousingType TEXT,
    MonthlyDebt REAL,
    ExistingAccounts INTEGER,
    CreatedAt TEXT NOT NULL,
    FOREIGN KEY (UserID) REFERENCES users (UserID)
);

CREATE TABLE IF NOT EXISTS credit_history (
    HistoryID INTEGER PRIMARY KEY AUTOINCREMENT,
    ApplicantID INTEGER,
    MonthsBalance INTEGER,
    PaymentStatus TEXT,
    OverdueStatus TEXT,
    FOREIGN KEY (ApplicantID) REFERENCES applicant_details (ApplicantID)
);

CREATE TABLE IF NOT EXISTS ml_model (
    ModelID INTEGER PRIMARY KEY AUTOINCREMENT,
    ModelName TEXT NOT NULL,
    AlgorithmType TEXT,
    Accuracy REAL,
    RocAuc REAL,
    ModelFile TEXT,
    IsActive INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS approval_prediction (
    PredictionID INTEGER PRIMARY KEY AUTOINCREMENT,
    ApplicantID INTEGER,
    ModelID INTEGER,
    ApprovalResult TEXT,
    RiskScore REAL,
    Reasons TEXT,
    PredictionDate TEXT NOT NULL,
    FOREIGN KEY (ApplicantID) REFERENCES applicant_details (ApplicantID),
    FOREIGN KEY (ModelID) REFERENCES ml_model (ModelID)
);
"""


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_conn() as conn:
        conn.executescript(SCHEMA)


def seed_ml_models(model_info_path="model/model_info.json"):
    """Populates ml_model from the trained-model metadata, if not already seeded."""
    with open(model_info_path) as f:
        info = json.load(f)

    with get_conn() as conn:
        existing = conn.execute("SELECT COUNT(*) AS n FROM ml_model").fetchone()["n"]
        if existing:
            return
        for name, metrics in info["results"].items():
            is_active = 1 if name == info["best_model_name"] else 0
            conn.execute(
                "INSERT INTO ml_model (ModelName, AlgorithmType, Accuracy, RocAuc, ModelFile, IsActive) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (name, name, metrics["accuracy"], metrics["roc_auc"],
                 "model/best_model.pkl" if is_active else None, is_active),
            )


def seed_demo_user():
    """Creates a fixed demo account (if it doesn't already exist) so anyone
    reviewing the project can log in immediately without registering."""
    from werkzeug.security import generate_password_hash
    if not get_user_by_email("demo@example.com"):
        create_user("Demo User", "demo@example.com", generate_password_hash("demo1234"))


# --- Users ------------------------------------------------------------------

def create_user(name, email, password_hash, role="analyst"):
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO users (Name, Email, Password, Role, CreatedAt) VALUES (?, ?, ?, ?, ?)",
            (name, email, password_hash, role, datetime.utcnow().isoformat()),
        )
        return cur.lastrowid


def get_user_by_email(email):
    with get_conn() as conn:
        return conn.execute("SELECT * FROM users WHERE Email = ?", (email,)).fetchone()


def get_user_by_id(user_id):
    with get_conn() as conn:
        return conn.execute("SELECT * FROM users WHERE UserID = ?", (user_id,)).fetchone()


# --- Applicants & predictions -------------------------------------------------

def save_applicant_and_prediction(user_id, row, model_name, approval_result, risk_score, reasons):
    """Stores the applicant profile + the resulting prediction, linked to the
    active ML model row. Returns (applicant_id, prediction_id)."""
    with get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO applicant_details
               (UserID, Age, Gender, AnnualIncome, CreditScore, EmploymentStatus,
                EmploymentYears, EducationType, FamilyStatus, HousingType,
                MonthlyDebt, ExistingAccounts, CreatedAt)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (user_id, row["AGE"], row["GENDER"], row["ANNUAL_INCOME"], row["CREDIT_SCORE"],
             row["EMPLOYMENT_STATUS"], row["EMPLOYMENT_YEARS"], row["EDUCATION_TYPE"],
             row["FAMILY_STATUS"], row["HOUSING_TYPE"], row["MONTHLY_DEBT"],
             row["EXISTING_ACCOUNTS"], datetime.utcnow().isoformat()),
        )
        applicant_id = cur.lastrowid

        model_row = conn.execute(
            "SELECT ModelID FROM ml_model WHERE ModelName = ?", (model_name,)
        ).fetchone()
        model_id = model_row["ModelID"] if model_row else None

        cur2 = conn.execute(
            """INSERT INTO approval_prediction
               (ApplicantID, ModelID, ApprovalResult, RiskScore, Reasons, PredictionDate)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (applicant_id, model_id, approval_result, risk_score,
             json.dumps(reasons), datetime.utcnow().isoformat()),
        )
        return applicant_id, cur2.lastrowid


def get_prediction_history(user_id, limit=25):
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT p.PredictionID, p.ApprovalResult, p.RiskScore, p.Reasons, p.PredictionDate,
                      a.Age, a.Gender, a.AnnualIncome, a.CreditScore, a.EmploymentStatus,
                      m.ModelName
               FROM approval_prediction p
               JOIN applicant_details a ON a.ApplicantID = p.ApplicantID
               LEFT JOIN ml_model m ON m.ModelID = p.ModelID
               WHERE a.UserID = ?
               ORDER BY p.PredictionDate DESC
               LIMIT ?""",
            (user_id, limit),
        ).fetchall()
        return rows
