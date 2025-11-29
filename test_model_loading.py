# test_model_loading.py
import joblib
import os

def test_model_loading():
    model_path = 'models/fraud_detector.pkl'
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found at: {model_path}")
        print("Please run: python models/train_model.py")
        return False
    
    try:
        model_data = joblib.load(model_path)
        print("✅ Model loaded successfully!")
        print(f"Model type: {type(model_data['model']).__name__}")
        print(f"Features: {model_data['feature_columns']}")
        print(f"Merchant columns: {model_data['merchant_columns']}")
        return True
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

if __name__ == "__main__":
    test_model_loading()