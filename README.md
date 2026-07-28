# Credit Card Approval Prediction

A hands-on machine learning project that predicts whether a credit card
application will be approved or rejected, built with Python, Flask,
scikit-learn/XGBoost, and a Flask web interface ("Ledger").

## 🌐 Live Demo

**https://credit-card-approval-prediction-w9n6.onrender.com**

*(Free tier — the first request after inactivity can take 30–60 seconds to wake up.)*

## Demo Video

🎥 https://drive.google.com/file/d/1IASstHlK-CoX6L2kvy_ofFda5zKZ3RiF/view?usp=sharing

## 👤 Demo Credentials

To try the app with a signed-in account (view Prediction History, etc.)
without registering:

- **Email:** `demo@example.com`
- **Password:** `demo1234`

Or register your own account — that works too.

## 📄 Project Documentation

See [`Project Documentation.pdf`](./Project%20Documentation.pdf) for a full
write-up: architecture, ER diagram, dataset/features, model evaluation, and
deployment notes.

Phase-wise design-thinking and planning documents (Brainstorming, Problem
Statement, Empathy Map, Requirement Analysis, Solution Architecture, Sprint
Planning, UAT) are in
[`Project Phase Wise/Phase Wise Templates/`](./Project%20Phase%20Wise/Phase%20Wise%20Templates).

## 📂 What's Inside

```text
credit_card_approval/

├── 1. Ideation Phase/
│   └── Brainstorming & Ideation Phase/
│       ├── Brainstorming - Idea Generation.docx
│       ├── Define Problem Statement Template.docx
│       └── Empathy Map Canvas.docx
│
├── 2. Requirement Analysis/
│   └── Requirement Analysis/
│       ├── Data Flow Diagrams and User Stories.docx
│       ├── Solution Requirements.docx
│       └── Technology Stack - Template.docx
│
├── 3. Project Design Phase/
│   └── Project Design Phase/
│       ├── Problem - Solution Fit Template/
│       ├── Proposed Solution/
│       └── Solution Architecture/
│
├── 4. Project Planning Phase/
│   └── Project Planning Phase/
│       └── Project Planning Template.docx
│
├── 5. Project Development Phase/
│   └── Project Developement/
│       ├── Performance Testing.docx
│       └── User Acceptance Testing FSD.docx
│
├── 6. Project Documentation/
│   └── Project Documentation.pdf
│
├── 7. Project Demonstration/
│   ├── Demo Video Link.txt
│   ├── Deployment Link.txt
│   └── GitHub Repository.txt
│
├── data/
├── model/
├── notebooks/
├── static/
├── templates/
├── app.py
├── db.py
├── preprocessing.py
├── train_models.py
├── generate_dataset.py
├── eda.py
├── deploy_to_watson_ml.py
├── requirements.txt
├── README.md
└── database.db
```

## 1. Environment setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Get the data

**Option A — use the real dataset (recommended):** download
`application_record.csv` and `credit_record.csv` from the Kaggle
"Credit Card Approval Prediction" dataset and place them in `data/`.

**Option B — use the bundled synthetic generator:**

```bash
python generate_dataset.py
```

This creates schema-accurate synthetic data so the rest of the pipeline runs
end-to-end without needing an internet connection or a Kaggle account.

## 3. Explore the data (optional)

```bash
python eda.py
```

Saves count plots and distribution plots to `notebooks/figures/`. Notebook
versions of the same analysis (`notebooks/eda.ipynb`) and model training
(`notebooks/model_training.ipynb`) are also included.

## 4. Train the models

```bash
python train_models.py
```

Trains Logistic Regression, Decision Tree, Random Forest, and XGBoost,
prints accuracy / ROC-AUC / confusion matrix / classification report for
each, and saves the best-performing model plus the fitted encoders and
scaler to `model/`.

## 5. Run the web app

```bash
python app.py
```

Open **http://127.0.0.1:5000**. The database (`database.db`) is created
automatically on first run, seeded with the 4 trained models' metadata.

- **Home** — model leaderboard
- **Apply** — enter an applicant profile, get an instant Approved/Rejected
  decision with a risk percentage and reasons
- **Sign up / Sign in** — optional; lets you save your applications
- **History** — past predictions for the signed-in user (requires an account)

You can use "Apply" without an account — it still works, it just won't be saved.

## Database schema (matches the project's ER diagram)

| Table | Purpose |
|---|---|
| `users` | UserID, Name, Email, Password (hashed), Role |
| `applicant_details` | One row per submitted application, linked to a user |
| `credit_history` | Reference table for the monthly payment-history data used in training (not populated per web request — see note below) |
| `ml_model` | Metadata for all 4 trained models (accuracy, ROC-AUC, which one is active) |
| `approval_prediction` | The result of each prediction, linked to both the applicant and the model that produced it |

**Note on `credit_history`:** this table exists in the schema to match the ER
diagram's description of monthly payment history, but it's populated from the
training dataset (`credit_record.csv`), not from live web submissions — a
single-point-in-time application form has no payment history to record yet.

## How the target label is built

Each applicant has a monthly payment history (`credit_record.csv`, `STATUS`
column: `0`–`5` for days-past-due buckets, `C` for paid off, `X` for no loan
that month). This project converts that history into a binary label: any
month with `STATUS` in `{2, 3, 4, 5}` (60+ days past due) marks the applicant
high-risk (`TARGET = 1`); otherwise they're in good standing (`TARGET = 0`).

## Hard eligibility rules (India)

Before any ML scoring, every application passes a hard eligibility gate
mirroring common Indian bank policy (these vary slightly by issuer — HDFC,
SBI Card, ICICI, Axis, etc. — but the pattern below is standard):

| Rule | Threshold |
|---|---|
| Minimum age | 18 (no card can be issued to a minor) |
| Independent income needed under 21 | Age 18–20 without income ~₹1.8L/year → secured/add-on card only |
| No independent income | Students/unemployed with ₹0 income → not eligible for an unsecured card |
| Minimum income — salaried | ₹1,80,000/year (~₹15,000/month) |
| Minimum income — self-employed | ₹2,40,000/year (ITR-proven) |
| Minimum CIBIL score | 650 (750+ preferred) |
| Maximum age — salaried | 60 |
| Maximum age — self-employed | 65 |

Applicants who fail any of these are rejected immediately with the specific
reason(s) shown — the ML model never even sees them, since these are policy
cutoffs, not statistical risk judgments.

Applicants who pass the gate are then scored by the ML model, which returns
Approved/Rejected, a risk percentage, and the top factors behind the decision
(CIBIL score, debt-to-income ratio, employment length, existing accounts).

## Features used

- CIBIL score (300–900)
- Annual income (₹)
- Debt-to-income ratio (derived from monthly debt payments)
- Employment status and length
- Number of existing credit accounts
- Age, Gender
- Education level
- Family status
- Housing type

## Notes on the synthetic data

`generate_dataset.py` builds applicant profiles with realistic correlations
(higher CIBIL score / lower debt-to-income / longer employment → lower risk)
so the trained models show sensible, explainable behavior. Income and credit
score are generated on Indian scales (₹ annual income, CIBIL 300–900) rather
than US-style FICO/USD.

## Deploying it publicly (GitHub + Render)

This repo is ready to push to GitHub and deploy on [Render](https://render.com)
(free tier, no credit card required):

1. **Push to GitHub:**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<repo-name>.git
   git push -u origin main
   ```

2. **Deploy on Render:**
   - Sign up at [render.com](https://render.com) (GitHub login works).
   - Click **New +** → **Web Service** → connect your GitHub repo.
   - Settings:
     - **Build Command:** `pip install -r requirements.txt`
     - **Start Command:** `gunicorn app:app`
     - **Instance Type:** Free
   - Click **Create Web Service**. Render builds and deploys automatically —
     you'll get a public URL like `https://your-app.onrender.com`.

The `Procfile` and `gunicorn` entry in `requirements.txt` are already set up
for this. `app.py` reads the `PORT` environment variable Render provides
automatically, so no code changes are needed.

**Note:** Render's free tier spins the app down after periods of inactivity,
so the first request after idling can take ~30–60 seconds to wake up — that's
normal, not a bug.

**Version pinning matters here.** `requirements.txt` pins exact versions
(numpy, pandas, scikit-learn, flask, werkzeug, gunicorn) and `.python-version`
pins Python 3.12 — matching exactly what was used to train and pickle
`model/best_model.pkl`. Unpickling a scikit-learn model with a different
scikit-learn/Python version than it was saved with can fail outright (e.g.
`ModuleNotFoundError: No module named '_loss'`), since scikit-learn's
internal module layout isn't guaranteed stable across versions. If you
retrain the model with a different scikit-learn version locally, update
these pins to match, or you'll hit the same error on deploy.

## IBM Cloud / Watson ML deployment

The project's architecture diagram specifies deployment via IBM Cloud and
Watson Machine Learning. `deploy_to_watson_ml.py` implements that path:
it registers `model/best_model.pkl` with the Watson ML repository and
creates a live scoring endpoint, using the current `ibm-watsonx-ai` SDK
(the older `ibm-watson-machine-learning` package is deprecated as of mid-2024).

**Status:** the code is written and its local logic (encoding/scaling a
sample request the same way `app.py` does) is tested, but it hasn't been run
against a live IBM Cloud account — IBM Cloud requires credit/debit card
verification even for its free "Lite" tier, which wasn't available. The
public, working deployment for this project is on Render (see above); this
script is ready to run the moment IBM Cloud access is available.

**To actually deploy it, once you have IBM Cloud access:**

1. Sign up at [cloud.ibm.com/registration](https://cloud.ibm.com/registration)
   (requires card verification, no charge within Lite tier limits).
2. In the **Catalog**, create a **Watson Machine Learning** service (Lite plan).
3. Create an **API key**: IBM Cloud console → **Manage** → **Access (IAM)** →
   **API keys** → **Create**. Copy the key (shown only once).
4. Create a **deployment space**: go to the **Deployments** section →
   **Spaces** → **New deployment space** → give it a name → **Create**.
   Once created, open it → **Manage** tab → copy the **Space ID**.
5. Fill in `IBM_CLOUD_API_KEY`, `WML_URL` (matches the region you picked —
   e.g. `https://us-south.ml.cloud.ibm.com`), and `DEPLOYMENT_SPACE_ID` at
   the top of `deploy_to_watson_ml.py`.
6. Run:
   ```bash
   pip install -r requirements-watson.txt
   python deploy_to_watson_ml.py
   ```
   This stores the model, deploys it, and scores one sample applicant to
   confirm the live endpoint works — printing the deployment ID and scoring
   URL at the end.
