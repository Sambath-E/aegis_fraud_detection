# dashboard/app.py
import streamlit as st
import pandas as pd
import requests
import time
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np

# Page configuration
st.set_page_config(
    page_title="AEGIS Fraud Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .fraud-alert {
        background-color: #ffcccc;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #ff0000;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<div class="main-header">🛡️ AEGIS - Real-time Fraud Detection Dashboard</div>', unsafe_allow_html=True)
st.markdown("Live monitoring system for detecting fraudulent transactions in real-time")

def check_api_health():
    """Check if the API is running"""
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def test_prediction():
    """Test the prediction endpoint with sample data"""
    test_data = {
        "transaction_id": int(datetime.now().timestamp() * 1000),
        "amount": 150.0,
        "time_of_day": 14,
        "day_of_week": 2,
        "merchant_category": "Online",
        "customer_history": 25,
        "avg_transaction_value": 85.0,
        "location_distance": 15.0
    }
    
    try:
        response = requests.post("http://localhost:8000/predict", json=test_data, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except:
        return None

def main():
    # Sidebar
    st.sidebar.title("Configuration")
    
    # API Configuration
    st.sidebar.subheader("API Settings")
    api_url = st.sidebar.text_input("API URL", "http://localhost:8000")
    auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)
    refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", 5, 60, 10)
    
    # Test Transaction Section
    st.sidebar.subheader("Test Transaction")
    with st.sidebar.form("test_transaction_form"):
        amount = st.number_input("Amount ($)", min_value=0.0, value=100.0, step=10.0)
        time_of_day = st.slider("Time of Day", 0, 23, 12)
        merchant = st.selectbox("Merchant Category", 
                               ["Retail", "Food", "Travel", "Online", "Entertainment"])
        location_distance = st.slider("Location Distance (km)", 0.0, 200.0, 10.0, step=5.0)
        customer_history = st.slider("Customer History", 1, 100, 25)
        
        test_button = st.form_submit_button("Test Transaction")
        
        if test_button:
            test_data = {
                "transaction_id": int(datetime.now().timestamp() * 1000),
                "amount": amount,
                "time_of_day": time_of_day,
                "day_of_week": datetime.now().weekday(),
                "merchant_category": merchant,
                "customer_history": customer_history,
                "avg_transaction_value": 85.0,
                "location_distance": location_distance
            }
            
            try:
                response = requests.post(f"{api_url}/predict", json=test_data, timeout=10)
                if response.status_code == 200:
                    result = response.json()
                    
                    # Initialize session state if not exists
                    if 'transactions' not in st.session_state:
                        st.session_state.transactions = []
                    
                    # Add timestamp and full data
                    result['timestamp'] = datetime.now()
                    result['test_data'] = test_data
                    st.session_state.transactions.append(result)
                    
                    # Show result
                    if result['is_fraud']:
                        st.sidebar.error(f"🚨 FRAUD DETECTED! Risk: {result['risk_level']}")
                    else:
                        st.sidebar.success(f"✅ Normal Transaction - Risk: {result['risk_level']}")
                        
                    st.sidebar.write(f"Probability: {result['fraud_probability']:.3f}")
                else:
                    st.sidebar.error("❌ API Error - Check if server is running")
            except Exception as e:
                st.sidebar.error(f"❌ Connection failed: {str(e)}")
    
    # Main Dashboard
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # API Status
        st.subheader("System Status")
        api_status = check_api_health()
        
        if api_status:
            st.success("✅ API is running")
            
            # Test model connection
            test_result = test_prediction()
            if test_result:
                st.success("✅ Model is loaded and responding")
            else:
                st.error("❌ Model not responding")
        else:
            st.error("❌ API is not running")
            st.info("Start the API with: `python api/app.py`")
    
    with col2:
        st.subheader("Quick Actions")
        if st.button("Generate Sample Data"):
            try:
                # This would call your data generator
                import subprocess
                result = subprocess.run(["python", "data/mock_data_generator.py"], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    st.success("✅ Sample data generated!")
                else:
                    st.error("❌ Error generating data")
            except:
                st.error("❌ Could not generate data")
        
        if st.button("Train Model"):
            try:
                import subprocess
                result = subprocess.run(["python", "models/train_model.py"], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    st.success("✅ Model trained successfully!")
                else:
                    st.error("❌ Error training model")
            except:
                st.error("❌ Could not train model")
    
    # Display transactions if any
    if 'transactions' in st.session_state and st.session_state.transactions:
        st.markdown("---")
        st.subheader("📊 Transaction History")
        
        transactions_df = pd.DataFrame(st.session_state.transactions)
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        total_tx = len(transactions_df)
        fraud_tx = transactions_df['is_fraud'].sum()
        high_risk = len(transactions_df[transactions_df['risk_level'] == 'High'])
        avg_probability = transactions_df['fraud_probability'].mean()
        
        with col1:
            st.metric("Total Transactions", total_tx)
        with col2:
            st.metric("Fraud Detected", fraud_tx)
        with col3:
            st.metric("High Risk", high_risk)
        with col4:
            st.metric("Avg Fraud Probability", f"{avg_probability:.3f}")
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Risk Level Distribution")
            risk_counts = transactions_df['risk_level'].value_counts()
            fig = px.pie(values=risk_counts.values, names=risk_counts.index,
                        color=risk_counts.index,
                        color_discrete_map={'High':'red', 'Medium':'orange', 'Low':'green'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Transactions by Merchant")
            merchant_counts = transactions_df['test_data'].apply(lambda x: x['merchant_category']).value_counts()
            fig = px.bar(x=merchant_counts.index, y=merchant_counts.values,
                        labels={'x': 'Merchant Category', 'y': 'Count'})
            st.plotly_chart(fig, use_container_width=True)
        
        # Recent transactions table
        st.subheader("Recent Transactions")
        display_data = []
        for tx in st.session_state.transactions[-10:]:  # Last 10 transactions
            display_data.append({
                'Timestamp': tx['timestamp'].strftime('%H:%M:%S'),
                'Amount': f"${tx['test_data']['amount']:.2f}",
                'Merchant': tx['test_data']['merchant_category'],
                'Risk Level': tx['risk_level'],
                'Fraud Probability': f"{tx['fraud_probability']:.3f}",
                'Is Fraud': '🚨 YES' if tx['is_fraud'] else '✅ NO'
            })
        
        if display_data:
            st.dataframe(pd.DataFrame(display_data), use_container_width=True)
    
    else:
        st.info("💡 No transactions yet. Use the sidebar to test a transaction.")
    
    # Instructions
    st.markdown("---")
    st.subheader("🚀 Getting Started")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **1. Start the API**
        ```bash
        python api/app.py
        ```
        """)
    
    with col2:
        st.markdown("""
        **2. Generate Data & Train Model**
        ```bash
        python data/mock_data_generator.py
        python models/train_model.py
        ```
        """)
    
    with col3:
        st.markdown("""
        **3. Test the System**
        - Use the sidebar to test transactions
        - Monitor real-time predictions
        - View fraud analytics
        """)
    
    # Auto-refresh
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()

if __name__ == "__main__":
    main()