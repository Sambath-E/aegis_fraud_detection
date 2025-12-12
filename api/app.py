# api/app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
import pandas as pd
import joblib
import numpy as np
from typing import List, Dict, Any
import os

# Global model variable
detector = None

class FraudDetector:
    """Lightweight wrapper for model operations in API"""
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_columns = []
        self.merchant_columns = []
    
    def load_model(self, filepath):
        """Load the trained model"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
        self.merchant_columns = model_data['merchant_columns']
        print(f"✅ Model loaded successfully with {len(self.feature_columns)} features")
        return self
    
    def prepare_features(self, df):
        """Prepare features for prediction"""
        # Create a copy to avoid modifying original data
        df_encoded = df.copy()
        
        # Convert categorical variables to dummy variables
        merchant_dummies = pd.get_dummies(df_encoded['merchant_category'], prefix='merchant_category')
        df_encoded = pd.concat([df_encoded, merchant_dummies], axis=1)
        
        # Ensure all expected merchant columns are present
        for col in self.merchant_columns:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
        
        # Select only the columns we need
        selected_columns = self.feature_columns + self.merchant_columns
        X = df_encoded[selected_columns]
        
        return X

# Lifespan event handler (replaces @app.on_event)
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup code
    global detector
    try:
        detector = FraudDetector()
        detector.load_model('models/fraud_detector.pkl')
        print("🎉 AEGIS Fraud Detection API is ready!")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        detector = None
    
    yield  # This is where the application runs
    
    # Shutdown code (if any)
    print("🛑 Shutting down AEGIS API...")

# Create FastAPI app with lifespan
app = FastAPI(
    title="AEGIS Fraud Detection API",
    description="Real-time transaction fraud detection system",
    version="1.0.0",
    lifespan=lifespan
)

class Transaction(BaseModel):
    transaction_id: int
    amount: float
    time_of_day: int
    day_of_week: int
    merchant_category: str
    customer_history: int
    avg_transaction_value: float
    location_distance: float

class PredictionResponse(BaseModel):
    transaction_id: int
    is_fraud: bool
    fraud_probability: float
    risk_level: str

@app.get("/")
async def root():
    return {
        "message": "AEGIS Fraud Detection API", 
        "status": "active",
        "version": "1.0.0"
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_fraud(transaction: Transaction):
    if detector is None or detector.model is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Please train the model first.")
    
    try:
        # Convert to DataFrame
        transaction_dict = transaction.dict()
        df = pd.DataFrame([transaction_dict])
        
        # Prepare features
        X = detector.prepare_features(df)
        X_scaled = detector.scaler.transform(X)
        
        # Predict
        prediction = detector.model.predict(X_scaled)[0]
        decision_score = detector.model.decision_function(X_scaled)[0]
        
        # Convert to probability-like score
        fraud_probability = 1 / (1 + np.exp(-decision_score))
        
        # Determine risk level
        if fraud_probability > 0.7:
            risk_level = "High"
        elif fraud_probability > 0.3:
            risk_level = "Medium"
        else:
            risk_level = "Low"
        
        print(f"🔍 Transaction {transaction.transaction_id}: {risk_level} risk (prob: {fraud_probability:.3f})")
        
        return PredictionResponse(
            transaction_id=transaction.transaction_id,
            is_fraud=prediction == -1,
            fraud_probability=round(fraud_probability, 4),
            risk_level=risk_level
        )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.get("/health")
async def health_check():
    model_status = "loaded" if detector and detector.model else "not loaded"
    return {
        "status": "healthy", 
        "model_status": model_status,
        "service": "AEGIS Fraud Detection API"
    }

@app.get("/model-info")
async def model_info():
    if detector is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    return {
        "feature_columns": detector.feature_columns,
        "merchant_columns": detector.merchant_columns,
        "model_type": type(detector.model).__name__,
        "features_count": len(detector.feature_columns) + len(detector.merchant_columns)
    }

# For running with uvicorn directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True)