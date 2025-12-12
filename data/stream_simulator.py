# data/stream_simulator.py
import time
import requests
import pandas as pd
import random
from datetime import datetime

class TransactionStreamSimulator:
    def __init__(self, api_url="http://localhost:8000/predict"):
        self.api_url = api_url
        self.df = pd.read_csv('data/raw/transactions.csv')
    
    def generate_live_transaction(self):
        """Generate a random transaction from dataset"""
        transaction = self.df.sample(1).iloc[0]
        
        return {
            "transaction_id": int(datetime.now().timestamp() * 1000),
            "amount": float(transaction['amount']),
            "time_of_day": int(transaction['time_of_day']),
            "day_of_week": int(transaction['day_of_week']),
            "merchant_category": transaction['merchant_category'],
            "customer_history": int(transaction['customer_history']),
            "avg_transaction_value": float(transaction['avg_transaction_value']),
            "location_distance": float(transaction['location_distance'])
        }
    
    def start_stream(self, interval=2):
        """Start simulating transaction stream"""
        print("Starting transaction stream...")
        try:
            while True:
                transaction = self.generate_live_transaction()
                
                # Send to API
                response = requests.post(self.api_url, json=transaction)
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"Transaction {result['transaction_id']}: "
                          f"Fraud: {result['is_fraud']}, "
                          f"Risk: {result['risk_level']}, "
                          f"Probability: {result['fraud_probability']}")
                else:
                    print(f"API Error: {response.status_code}")
                
                time.sleep(interval)
        
        except KeyboardInterrupt:
            print("Stream stopped.")

if __name__ == "__main__":
    simulator = TransactionStreamSimulator()
    simulator.start_stream(interval=1)