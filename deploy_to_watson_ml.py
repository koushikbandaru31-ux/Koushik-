"""
deploy_to_watson_ml.py
-----------------------
Registers and deploys the trained credit card approval model to IBM
watsonx.ai / Watson Machine Learning, completing the "Machine Learning
Layer -> Model Storage -> IBM Cloud" path in the project's architecture
diagram.

This script is NOT run automatically -- it needs your own IBM Cloud
credentials (see README.md, "IBM Cloud / Watson ML deployment" section, for
how to get them). Once you have them, fill in the 4 values below and run:

    pip install ibm-watsonx-ai
    python deploy_to_watson_ml.py

What it does:
  1. Authenticates to watsonx.ai with your API key
  2. Uploads model/best_model.pkl to the Watson ML model repository
  3. Creates an online deployment (a live scoring endpoint)
  4. Sends one test applicant profile to the deployed endpoint and prints
     the prediction, to confirm it's working end-to-end
"""

import json
import pickle

from ibm_watsonx_ai import APIClient, Credentials

# --- 1. Fill these in from your IBM Cloud account ---------------------------
IBM_CLOUD_API_KEY = "<YOUR_IBM_CLOUD_API_KEY>"        # IAM > API keys > Create
WML_URL = "https://us-south.ml.cloud.ibm.com"          # match your service's region
DEPLOYMENT_SPACE_ID = "<YOUR_DEPLOYMENT_SPACE_ID>"     # Deployments > Spaces > (create one) > Manage tab

with open("model/best_model.pkl", "rb") as f:
    MODEL = pickle.load(f)

with open("model/model_info.json") as f:
    MODEL_INFO = json.load(f)


def main():
    credentials = Credentials(url=WML_URL, api_key=IBM_CLOUD_API_KEY)
    client = APIClient(credentials)
    client.set.default_space(DEPLOYMENT_SPACE_ID)

    # Match the software spec to whatever's current in your account --
    # list options with: client.software_specifications.list()
    sw_spec_id = client.software_specifications.get_id_by_name("runtime-24.1-py3.11")

    print(f"Storing model '{MODEL_INFO['best_model_name']}' in the WML repository...")
    model_meta = {
        client.repository.ModelMetaNames.NAME: "credit-card-approval-model",
        client.repository.ModelMetaNames.TYPE: "scikit-learn_1.3",
        client.repository.ModelMetaNames.SOFTWARE_SPEC_ID: sw_spec_id,
    }
    stored_model = client.repository.store_model(model=MODEL, meta_props=model_meta)
    model_id = client.repository.get_model_id(stored_model)
    print(f"Model stored. Model ID: {model_id}")

    print("Creating online deployment...")
    deployment_meta = {
        client.deployments.ConfigurationMetaNames.NAME: "credit-card-approval-deployment",
        client.deployments.ConfigurationMetaNames.ONLINE: {},
    }
    deployment = client.deployments.create(model_id, meta_props=deployment_meta)
    deployment_id = client.deployments.get_id(deployment)
    print(f"Deployed. Deployment ID: {deployment_id}")

    # --- Test the live endpoint with one sample applicant -------------------
    # Note: this deploys the raw classifier, not a full pipeline, so inputs
    # must be pre-encoded/scaled exactly like preprocessing.py does for local
    # predictions in app.py. We load the same encoders/scaler here to build
    # a correctly formatted sample request.
    with open("model/encoders.pkl", "rb") as f:
        encoders = pickle.load(f)
    with open("model/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    from preprocessing import FEATURE_COLS, CATEGORICAL_COLS
    import numpy as np

    sample_row = {
        "AGE": 35, "ANNUAL_INCOME": 1200000, "CREDIT_SCORE": 810,
        "EMPLOYMENT_YEARS": 10, "MONTHLY_DEBT": 8000,
        "DEBT_TO_INCOME": (8000 * 12) / 1200000, "EXISTING_ACCOUNTS": 1,
        "EMPLOYMENT_STATUS": "Salaried", "GENDER": "F",
        "EDUCATION_TYPE": "Post Graduate", "FAMILY_STATUS": "Married",
        "HOUSING_TYPE": "Owned",
    }
    encoded_values = []
    for col in FEATURE_COLS:
        val = sample_row[col]
        if col in CATEGORICAL_COLS:
            val = encoders[col].transform([val])[0]
        encoded_values.append(float(val))
    scaled_values = scaler.transform(np.array(encoded_values).reshape(1, -1))[0].tolist()

    scoring_payload = {
        client.deployments.ScoringMetaNames.INPUT_DATA: [{
            "fields": FEATURE_COLS,
            "values": [scaled_values],
        }]
    }
    result = client.deployments.score(deployment_id, scoring_payload)
    print("Sample scoring result:")
    print(json.dumps(result, indent=2))

    print(f"\nPublic scoring endpoint URL pattern:\n"
          f"{WML_URL}/ml/v4/deployments/{deployment_id}/predictions?version=2021-05-01")


if __name__ == "__main__":
    main()
