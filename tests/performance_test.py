# tests/performance_test.py
import requests
import time
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

def test_single_prediction():
    """Test single prediction performance"""
    test_data = {
        "transaction_id": 1,
        "amount": 100.0,
        "time_of_day": 12,
        "day_of_week": 1,
        "merchant_category": "Retail",
        "customer_history": 10,
        "avg_transaction_value": 50.0,
        "location_distance": 5.0
    }
    
    start_time = time.time()
    response = requests.post("http://localhost:8000/predict", json=test_data)
    end_time = time.time()
    
    return {
        "status_code": response.status_code,
        "response_time": end_time - start_time,
        "success": response.status_code == 200
    }

def load_test(num_requests=100):
    """Perform load testing"""
    print(f"Starting load test with {num_requests} requests...")
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(lambda x: test_single_prediction(), range(num_requests)))
    
    successful_requests = sum(1 for r in results if r['success'])
    response_times = [r['response_time'] for r in results if r['success']]
    
    print(f"Success Rate: {successful_requests/num_requests*100:.1f}%")
    print(f"Average Response Time: {sum(response_times)/len(response_times)*1000:.2f}ms")
    print(f"Max Response Time: {max(response_times)*1000:.2f}ms")
    print(f"Min Response Time: {min(response_times)*1000:.2f}ms")

if __name__ == "__main__":
    load_test(100)