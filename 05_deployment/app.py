from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os
import uvicorn

# Define paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, 'artifacts')
MODEL_PATH = os.path.join(ARTIFACTS_DIR, 'lgbm_model.pkl')
FEATURES_PATH = os.path.join(ARTIFACTS_DIR, 'features.joblib')

app = FastAPI(title="Credit Default Risk API", description="API to predict credit default risk using LightGBM.")

# Global variables for model and features
model = None
features = None

@app.on_event("startup")
def load_artifacts():
    global model, features
    try:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            print(f"Model loaded from {MODEL_PATH}")
        else:
            print(f"Warning: Model not found at {MODEL_PATH}")

        if os.path.exists(FEATURES_PATH):
            features = joblib.load(FEATURES_PATH)
            print(f"Features list loaded from {FEATURES_PATH}")
        else:
            print(f"Warning: Features list not found at {FEATURES_PATH}")
            
    except Exception as e:
        print(f"Error loading artifacts: {e}")

class ApplicantData(BaseModel):
    # Accepting a dictionary because there are hundreds of features
    data: dict

@app.get("/")
def read_root():
    return {"message": "Credit Default Risk API is running."}

@app.post("/evaluate_risk")
def evaluate_risk(applicant: ApplicantData):
    if model is None or features is None:
        raise HTTPException(status_code=503, detail="Model or features not loaded.")
    
    try:
        # Sanitize input keys to match training features (same logic as in train.py)
        sanitized_data = {}
        for k, v in applicant.data.items():
            clean_k = "".join(c if c.isalnum() else "_" for c in str(k))
            sanitized_data[clean_k] = v

        # Convert input dict to DataFrame
        input_data = pd.DataFrame([sanitized_data])
        
        # Align with training features
        # Create a DataFrame with all training features initialized to NaN (or 0)
        # Efficient way: Reindex
        input_data_aligned = input_data.reindex(columns=features)
        
        # Prediction
        prob = model.predict(input_data_aligned)[0]
        
        # Decision Logic
        decision = "REVISIÓN MANUAL"
        if prob < 0.08:
            decision = "APROBAR"
        elif prob > 0.3: # Thresholds should be tuned based on business requirements
            decision = "RECHAZAR"
            
        return {
            "default_probability": float(prob),
            "decision": decision,
            "risk_level": "High" if prob > 0.5 else ("Medium" if prob > 0.1 else "Low")
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
