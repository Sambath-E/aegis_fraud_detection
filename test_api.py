# test_api.py
import requests
import json

# Test data
test_transaction = {
    "transaction_id": 10001,
    "amount": 1500.0,
    "time_of_day": 3,
    "day_of_week": 2,
    "merchant_category": "Online",
    "customer_history": 15,
    "avg_transaction_value": 120.0,
    "location_distance": 150.0
}

# Make prediction
response = requests.post("http://localhost:8000/predict", json=test_transaction)
print("Response:", json.dumps(response.json(), indent=2))