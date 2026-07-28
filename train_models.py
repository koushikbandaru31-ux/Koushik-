"""
train_models.py
----------------
Step 5 of the project: Machine Learning Model Building.

Trains four classification algorithms:
  - Logistic Regression
  - Decision Tree Classifier
  - Random Forest Classifier
  - XGBoost Classifier (Gradient Boosting)

Evaluates each with accuracy, confusion matrix, and classification report,
then saves the best-performing model (plus the fitted encoders/scaler) to
model/ for use by the Flask app.
"""

import json
import pickle

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    # Falls back to an equivalent gradient-boosting model if xgboost isn't
    # installed in this environment. Install `xgboost` for the real thing --
    # it's listed in requirements.txt.
    from sklearn.ensemble import GradientBoostingClassifier as XGBClassifier
    HAS_XGB = False

from preprocessing import FEATURE_COLS, build_dataset

MODEL_DIR = "model"


def get_models():
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=10, random_state=42, n_jobs=-1
        ),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.08,
            eval_metric="logloss", random_state=42, use_label_encoder=False,
        )
    else:
        models["XGBoost"] = XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.08, random_state=42
        )
    return models


def main():
    X_scaled, y, encoders, scaler, df = build_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    results = {}
    fitted_models = {}

    for name, model in get_models().items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, preds)
        auc = roc_auc_score(y_test, proba)
        cm = confusion_matrix(y_test, preds).tolist()
        report = classification_report(y_test, preds, output_dict=True)

        results[name] = {"accuracy": acc, "roc_auc": auc, "confusion_matrix": cm}
        fitted_models[name] = model

        print(f"\n=== {name} ===")
        print(f"Accuracy: {acc:.4f}   ROC-AUC: {auc:.4f}")
        print(f"Confusion Matrix: {cm}")
        print(classification_report(y_test, preds))

    best_name = max(results, key=lambda k: results[k]["roc_auc"])
    best_model = fitted_models[best_name]
    print(f"\n>>> Best model: {best_name} (ROC-AUC={results[best_name]['roc_auc']:.4f})")

    # Persist best model + preprocessing artifacts for the Flask app
    with open(f"{MODEL_DIR}/best_model.pkl", "wb") as f:
        pickle.dump(best_model, f)
    with open(f"{MODEL_DIR}/encoders.pkl", "wb") as f:
        pickle.dump(encoders, f)
    with open(f"{MODEL_DIR}/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(f"{MODEL_DIR}/feature_cols.json", "w") as f:
        json.dump(FEATURE_COLS, f)
    with open(f"{MODEL_DIR}/model_info.json", "w") as f:
        json.dump({
            "best_model_name": best_name,
            "results": {k: {"accuracy": v["accuracy"], "roc_auc": v["roc_auc"]}
                        for k, v in results.items()},
        }, f, indent=2)

    print(f"\nSaved best model ({best_name}) and preprocessing artifacts to {MODEL_DIR}/")


if __name__ == "__main__":
    main()
