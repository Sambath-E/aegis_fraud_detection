# data/mock_data_generator.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

def generate_transaction_data(num_records=10000):
    """Generate realistic mock transaction data with fraud patterns"""
    np.random.seed(42)
    
    # Create base transaction data
    data = {
        'transaction_id': range(1, num_records + 1),
        'amount': np.random.exponential(100, num_records),
        'time_of_day': np.random.randint(0, 24, num_records),
        'day_of_week': np.random.randint(0, 7, num_records),
        'merchant_category': np.random.choice(['Retail', 'Food', 'Travel', 'Online', 'Entertainment'], num_records),
        'customer_history': np.random.randint(1, 100, num_records),
        'avg_transaction_value': np.random.exponential(50, num_records),
        'location_distance': np.random.exponential(10, num_records),
    }
    
    df = pd.DataFrame(data)
    
    # Create fraud labels (3% fraud rate)
    fraud_indices = np.random.choice(num_records, size=int(num_records * 0.03), replace=False)
    
    # Initialize fraud column
    df['is_fraud'] = 0
    df.loc[fraud_indices, 'is_fraud'] = 1
    
    # Add realistic fraud patterns to fraud transactions
    fraud_mask = df['is_fraud'] == 1
    
    # Fraud transactions are typically larger
    df.loc[fraud_mask, 'amount'] = df.loc[fraud_mask, 'amount'] * np.random.uniform(2, 5, fraud_mask.sum())
    
    # Fraud often happens from unusual locations
    df.loc[fraud_mask, 'location_distance'] = df.loc[fraud_mask, 'location_distance'] * np.random.uniform(3, 8, fraud_mask.sum())
    
    # Fraud often happens at unusual hours (late night/early morning)
    df.loc[fraud_mask, 'time_of_day'] = np.random.choice([0, 1, 2, 3, 4, 5, 22, 23], fraud_mask.sum())
    
    # Fraud often involves new customers or unusual merchant categories
    df.loc[fraud_mask, 'customer_history'] = np.random.randint(1, 10, fraud_mask.sum())
    df.loc[fraud_mask, 'merchant_category'] = np.random.choice(['Online', 'Travel'], fraud_mask.sum())
    
    # Add some timestamp data
    start_date = datetime(2024, 1, 1)
    df['timestamp'] = [start_date + timedelta(hours=i) for i in range(num_records)]
    
    return df

def save_data(df, filepath):
    """Save data to CSV"""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    df.to_csv(filepath, index=False)
    print(f"✅ Data saved to {filepath}")
    print(f"📊 Dataset info: {len(df)} transactions, {df['is_fraud'].sum()} fraud cases ({df['is_fraud'].mean()*100:.1f}%)")

def analyze_data(df):
    """Print basic data analysis"""
    print("\n📈 Data Analysis:")
    print(f"Total transactions: {len(df):,}")
    print(f"Fraud cases: {df['is_fraud'].sum():,} ({df['is_fraud'].mean()*100:.2f}%)")
    print(f"Average amount: ${df['amount'].mean():.2f}")
    print(f"Average fraud amount: ${df[df['is_fraud']==1]['amount'].mean():.2f}")
    print(f"Average normal amount: ${df[df['is_fraud']==0]['amount'].mean():.2f}")
    
    print("\n🛍️ Merchant Category Distribution:")
    print(df['merchant_category'].value_counts())
    
    print("\n⏰ Time of Day Patterns:")
    print("Fraud transactions by hour:")
    fraud_by_hour = df[df['is_fraud']==1]['time_of_day'].value_counts().sort_index()
    print(fraud_by_hour)

# Generate and save data
if __name__ == "__main__":
    print("🎲 Generating mock transaction data...")
    
    # Generate data
    df = generate_transaction_data(10000)
    
    # Analyze data
    analyze_data(df)
    
    # Save data
    save_data(df, 'data/raw/transactions.csv')
    
    print("\n✅ Data generation completed successfully!")
    print("📁 File saved: data/raw/transactions.csv")