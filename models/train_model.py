# models/train_model.py
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

class FraudDetector:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = ['amount', 'time_of_day', 'day_of_week', 
                               'customer_history', 'avg_transaction_value', 'location_distance']
        self.merchant_columns = []
    
    def prepare_features(self, df):
        """Prepare features for training/prediction"""
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
    
    def train_isolation_forest(self, df):
        """Train Isolation Forest model for fraud detection"""
        print("🔧 Preparing features...")
        
        # First, determine merchant columns from training data
        merchant_dummies = pd.get_dummies(df['merchant_category'], prefix='merchant_category')
        self.merchant_columns = merchant_dummies.columns.tolist()
        
        X = self.prepare_features(df)
        y = df['is_fraud']
        
        print(f"📊 Using {len(self.feature_columns)} base features and {len(self.merchant_columns)} merchant features")
        print(f"🎯 Fraud rate in training data: {y.mean():.3f}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        print(f"📈 Training set: {len(X_train):,} transactions")
        print(f"📊 Test set: {len(X_test):,} transactions")
        
        # Scale features
        print("⚖️ Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        print("🤖 Training Isolation Forest model...")
        self.model = IsolationForest(
            contamination=0.03,  # Expected fraud rate
            random_state=42,
            n_estimators=100,
            max_samples=256
        )
        
        self.model.fit(X_train_scaled)
        
        # Evaluate model
        print("📊 Evaluating model performance...")
        self.evaluate_model(X_test_scaled, y_test)
        
        return self.model
    
    def evaluate_model(self, X_test_scaled, y_test):
        """Evaluate model performance"""
        # Predict on test set
        test_predictions = self.model.predict(X_test_scaled)
        test_scores = self.model.decision_function(X_test_scaled)
        
        # Convert to binary (1 for fraud, 0 for normal)
        # Isolation Forest returns -1 for anomalies (fraud), 1 for normal
        test_predictions_binary = [1 if x == -1 else 0 for x in test_predictions]
        
        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, test_predictions_binary, average='binary', zero_division=0
        )
        
        print("\n" + "="*50)
        print("🎯 MODEL PERFORMANCE METRICS")
        print("="*50)
        print(f"📍 Precision: {precision:.3f}")
        print(f"📍 Recall:    {recall:.3f}")
        print(f"📍 F1-Score:  {f1:.3f}")
        
        print("\n📋 Classification Report:")
        print(classification_report(y_test, test_predictions_binary, target_names=['Normal', 'Fraud']))
        
        print("📊 Confusion Matrix:")
        cm = confusion_matrix(y_test, test_predictions_binary)
        print(cm)
        
        # Calculate accuracy
        accuracy = (test_predictions_binary == y_test).mean()
        print(f"📍 Accuracy:  {accuracy:.3f}")
        
        return test_predictions_binary, test_scores
    
    def save_model(self, filepath):
        """Save trained model and all necessary components"""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'merchant_columns': self.merchant_columns
        }
        joblib.dump(model_data, filepath)
        print(f"💾 Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load trained model and all components"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
        self.merchant_columns = model_data['merchant_columns']
        print(f"📂 Model loaded from {filepath}")
        return self

def check_data_quality(df):
    """Check if data is suitable for training"""
    print("\n🔍 Data Quality Check:")
    print(f"Total records: {len(df):,}")
    print(f"Fraud cases: {df['is_fraud'].sum():,}")
    print(f"Fraud rate: {df['is_fraud'].mean():.3f}")
    
    # Check for missing values
    missing = df.isnull().sum()
    if missing.any():
        print("❌ Missing values found:")
        print(missing[missing > 0])
        return False
    else:
        print("✅ No missing values")
    
    # Check if we have enough fraud cases
    if df['is_fraud'].sum() < 10:
        print("❌ Not enough fraud cases for training")
        return False
    
    # Check feature columns
    required_columns = ['amount', 'time_of_day', 'day_of_week', 'merchant_category', 
                       'customer_history', 'avg_transaction_value', 'location_distance', 'is_fraud']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"❌ Missing columns: {missing_columns}")
        return False
    
    print("✅ Data quality check passed")
    return True

# Main execution
if __name__ == "__main__":
    print("🚀 Starting Fraud Detection Model Training...")
    
    # Check if data file exists
    data_path = 'data/raw/transactions.csv'
    if not os.path.exists(data_path):
        print(f"❌ Data file not found: {data_path}")
        print("Please run: python data/mock_data_generator.py")
        exit(1)
    
    # Load data
    print("📂 Loading transaction data...")
    df = pd.read_csv(data_path)
    print(f"✅ Loaded {len(df):,} transactions")
    
    # Check data quality
    if not check_data_quality(df):
        print("❌ Data quality check failed. Please generate new data.")
        exit(1)
    
    # Train model
    detector = FraudDetector()
    detector.train_isolation_forest(df)
    
    # Save model
    detector.save_model('models/fraud_detector.pkl')
    
    print("\n🎉 Model training completed successfully!")
    print("📁 Model saved: models/fraud_detector.pkl")
    print("🔧 You can now start the API: python api/app.py")