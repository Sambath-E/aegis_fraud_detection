# test_model.py
import joblib
import pandas as pd
import os

def test_model():
    model_path = 'models/fraud_detector.pkl'
    
    if not os.path.exists(model_path):
        print("❌ Model file not found. Please train the model first.")
        return False
    
    try:
        # Load the model
        model_data = joblib.load(model_path)
        print("✅ Model loaded successfully!")
        print(f"Model type: {type(model_data['model']).__name__}")
        print(f"Feature columns: {model_data['feature_columns']}")
        print(f"Merchant columns: {model_data['merchant_columns']}")
        
        # Test prediction with sample data
        detector = type('MockDetector', (), {})()
        detector.model = model_data['model']
        detector.scaler = model_data['scaler']
        detector.feature_columns = model_data['feature_columns']
        detector.merchant_columns = model_data['merchant_columns']
        
        # Create test transaction
        test_data = {
            'amount': 150.0,
            'time_of_day': 2,
            'day_of_week': 1,
            'merchant_category': 'Online',
            'customer_history': 10,
            'avg_transaction_value': 85.0,
            'location_distance': 25.0
        }
        
        df_test = pd.DataFrame([test_data])
        
        # Prepare features (simplified)
        merchant_dummies = pd.get_dummies(df_test['merchant_category'], prefix='merchant_category')
        df_encoded = pd.concat([df_test, merchant_dummies], axis=1)
        
        # Ensure all merchant columns are present
        for col in detector.merchant_columns:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
        
        selected_columns = detector.feature_columns + detector.merchant_columns
        X_test = df_encoded[selected_columns]
        X_test_scaled = detector.scaler.transform(X_test)
        
        # Make prediction
        prediction = detector.model.predict(X_test_scaled)[0]
        score = detector.model.decision_function(X_test_scaled)[0]
        
        print(f"🎯 Prediction: {'Fraud' if prediction == -1 else 'Normal'}")
        print(f"📊 Anomaly score: {score:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    test_model()