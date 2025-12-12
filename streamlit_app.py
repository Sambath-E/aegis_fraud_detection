# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time
import sys
import os

# Add project directories to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Page configuration
st.set_page_config(
    page_title="AEGIS Fraud Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
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

# Title
st.markdown('<div class="main-header">🛡️ AEGIS - Real-time Fraud Detection Dashboard</div>', unsafe_allow_html=True)
st.markdown("""
**Cloud Deployment Demo** - This is a simulated version of the AEGIS fraud detection system.
For full functionality with FastAPI backend, run locally.
""")

# Initialize session state
if 'transactions' not in st.session_state:
    st.session_state.transactions = []
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False

def generate_sample_data():
    """Generate sample transaction data for demo"""
    np.random.seed(42)
    
    data = {
        'timestamp': pd.date_range(start='2024-01-01', periods=100, freq='H'),
        'amount': np.random.exponential(100, 100),
        'merchant': np.random.choice(['Retail', 'Food', 'Travel', 'Online', 'Entertainment'], 100),
        'location': np.random.choice(['Local', 'Regional', 'National', 'International'], 100),
        'time_of_day': np.random.randint(0, 24, 100),
    }
    
    df = pd.DataFrame(data)
    
    # Simulate fraud (5% of transactions)
    fraud_indices = np.random.choice(100, size=5, replace=False)
    df['is_fraud'] = False
    df.loc[fraud_indices, 'is_fraud'] = True
    
    # Make fraud transactions look different
    df.loc[fraud_indices, 'amount'] = df.loc[fraud_indices, 'amount'] * np.random.uniform(3, 10, 5)
    df.loc[fraud_indices, 'location'] = 'International'
    df.loc[fraud_indices, 'time_of_day'] = np.random.choice([0, 1, 2, 3, 22, 23], 5)
    
    return df

def simulate_fraud_prediction(transaction):
    """Simulate ML model prediction for cloud demo"""
    # Simplified fraud detection logic
    risk_score = 0
    
    # Amount-based risk
    if transaction['amount'] > 500:
        risk_score += 0.3
    elif transaction['amount'] > 1000:
        risk_score += 0.5
    
    # Time-based risk (late night)
    if transaction['time_of_day'] < 6 or transaction['time_of_day'] > 22:
        risk_score += 0.2
    
    # Location-based risk
    if transaction['location'] == 'International':
        risk_score += 0.3
    
    # Add some randomness
    risk_score += np.random.uniform(-0.1, 0.1)
    
    # Ensure score is between 0 and 1
    risk_score = max(0, min(1, risk_score))
    
    # Determine risk level
    if risk_score > 0.7:
        risk_level = "High"
        is_fraud = np.random.choice([True, False], p=[0.8, 0.2])
    elif risk_score > 0.3:
        risk_level = "Medium"
        is_fraud = np.random.choice([True, False], p=[0.3, 0.7])
    else:
        risk_level = "Low"
        is_fraud = False
    
    return {
        'is_fraud': is_fraud,
        'risk_level': risk_level,
        'fraud_probability': round(risk_score, 3)
    }

def main():
    # Sidebar
    st.sidebar.title("⚙️ Configuration")
    
    # Demo Mode Selection
    demo_mode = st.sidebar.selectbox(
        "Select Mode",
        ["Live Simulation", "Historical Analysis", "Model Training Demo"]
    )
    
    # Data Controls
    st.sidebar.subheader("📊 Data Controls")
    num_transactions = st.sidebar.slider("Number of transactions", 10, 500, 100)
    fraud_rate = st.sidebar.slider("Fraud rate (%)", 1, 10, 3) / 100
    
    # Auto-refresh
    auto_refresh = st.sidebar.checkbox("Auto-refresh", value=False)
    refresh_rate = st.sidebar.slider("Refresh rate (seconds)", 5, 60, 10)
    
    # Generate Data Button
    if st.sidebar.button("🔄 Generate Sample Data", type="primary"):
        with st.spinner("Generating transaction data..."):
            time.sleep(2)  # Simulate processing time
            st.session_state.sample_data = generate_sample_data()
            st.sidebar.success(f"Generated {len(st.session_state.sample_data)} transactions!")
    
    # Main Content Area
    if demo_mode == "Live Simulation":
        st.header("🎯 Live Fraud Detection Simulation")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Test a Transaction")
            
            with st.form("transaction_form"):
                amount = st.number_input("Amount ($)", min_value=0.0, value=100.0, step=10.0)
                merchant = st.selectbox("Merchant Category", 
                                      ["Retail", "Food", "Travel", "Online", "Entertainment"])
                location = st.selectbox("Location", 
                                      ["Local", "Regional", "National", "International"])
                time_of_day = st.slider("Time of Day", 0, 23, 12)
                
                submitted = st.form_submit_button("🔍 Check for Fraud")
                
                if submitted:
                    transaction = {
                        'transaction_id': int(datetime.now().timestamp() * 1000),
                        'amount': amount,
                        'merchant': merchant,
                        'location': location,
                        'time_of_day': time_of_day,
                        'timestamp': datetime.now()
                    }
                    
                    # Get prediction
                    prediction = simulate_fraud_prediction(transaction)
                    
                    # Store in session
                    result = {**transaction, **prediction}
                    st.session_state.transactions.append(result)
                    
                    # Display result
                    if prediction['is_fraud']:
                        st.error(f"🚨 **FRAUD DETECTED!**")
                        st.error(f"Risk Level: **{prediction['risk_level']}**")
                    else:
                        st.success(f"✅ **Transaction Normal**")
                        st.info(f"Risk Level: **{prediction['risk_level']}**")
                    
                    st.metric("Fraud Probability", f"{prediction['fraud_probability']:.1%}")
        
        with col2:
            st.subheader("Quick Actions")
            if st.button("🚀 Simulate 10 Transactions"):
                with st.spinner("Processing..."):
                    for _ in range(10):
                        transaction = {
                            'transaction_id': int(datetime.now().timestamp() * 1000) + _,
                            'amount': np.random.exponential(100),
                            'merchant': np.random.choice(['Retail', 'Food', 'Travel', 'Online', 'Entertainment']),
                            'location': np.random.choice(['Local', 'Regional', 'National', 'International']),
                            'time_of_day': np.random.randint(0, 24),
                            'timestamp': datetime.now()
                        }
                        prediction = simulate_fraud_prediction(transaction)
                        result = {**transaction, **prediction}
                        st.session_state.transactions.append(result)
                    st.success("10 transactions simulated!")
            
            if st.button("📊 View Statistics"):
                if 'sample_data' in st.session_state:
                    st.subheader("Dataset Statistics")
                    df = st.session_state.sample_data
                    st.write(f"Total transactions: {len(df)}")
                    st.write(f"Fraud cases: {df['is_fraud'].sum()} ({df['is_fraud'].mean()*100:.1f}%)")
                    st.write(f"Average amount: ${df['amount'].mean():.2f}")
                else:
                    st.warning("Generate sample data first!")
    
    elif demo_mode == "Historical Analysis":
        st.header("📈 Historical Analysis")
        
        if 'sample_data' in st.session_state:
            df = st.session_state.sample_data
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Transactions", len(df))
            with col2:
                st.metric("Fraud Cases", df['is_fraud'].sum())
            with col3:
                st.metric("Fraud Rate", f"{df['is_fraud'].mean()*100:.1f}%")
            with col4:
                st.metric("Avg Amount", f"${df['amount'].mean():.2f}")
            
            # Charts
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Fraud by Time of Day")
                fraud_by_hour = df[df['is_fraud']].groupby('time_of_day').size()
                fig = px.bar(x=fraud_by_hour.index, y=fraud_by_hour.values,
                           labels={'x': 'Hour of Day', 'y': 'Fraud Count'})
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Fraud by Merchant")
                fraud_by_merchant = df[df['is_fraud']]['merchant'].value_counts()
                fig = px.pie(values=fraud_by_merchant.values, 
                           names=fraud_by_merchant.index)
                st.plotly_chart(fig, use_container_width=True)
            
            # Data Table
            st.subheader("Transaction Data")
            st.dataframe(df.head(20), use_container_width=True)
        else:
            st.info("Click 'Generate Sample Data' in the sidebar to see historical analysis.")
    
    else:  # Model Training Demo
        st.header("🤖 Model Training Demo")
        
        st.info("""
        **Note:** Streamlit Cloud doesn't support FastAPI servers or running separate ML training.
        This demo simulates the training process.
        """)
        
        if st.button("🎯 Train Model (Simulated)"):
            with st.spinner("Training Isolation Forest model..."):
                progress_bar = st.progress(0)
                
                for i in range(100):
                    time.sleep(0.02)  # Simulate training time
                    progress_bar.progress(i + 1)
                
                st.session_state.model_trained = True
                st.success("✅ Model trained successfully!")
            
            st.markdown("""
            **Model Performance (Simulated):**
            - Accuracy: 96.2%
            - Precision: 28.7%
            - Recall: 27.4%
            - F1-Score: 0.280
            
            **Model Details:**
            - Algorithm: Isolation Forest
            - Features: 11 engineered features
            - Training samples: 7,000
            - Test samples: 3,000
            """)
    
    # Display recent transactions if any
    if st.session_state.transactions:
        st.markdown("---")
        st.subheader("📋 Recent Transactions")
        
        transactions_df = pd.DataFrame(st.session_state.transactions)
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Tested", len(transactions_df))
        with col2:
            fraud_count = transactions_df['is_fraud'].sum()
            st.metric("Fraud Detected", fraud_count)
        with col3:
            if len(transactions_df) > 0:
                fraud_rate = fraud_count / len(transactions_df)
                st.metric("Detection Rate", f"{fraud_rate:.1%}")
        
        # Display table
        display_df = transactions_df.tail(10)[['timestamp', 'amount', 'merchant', 'risk_level', 'fraud_probability', 'is_fraud']]
        display_df['is_fraud'] = display_df['is_fraud'].apply(lambda x: '🚨 YES' if x else '✅ NO')
        st.dataframe(display_df, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **💡 For Full Functionality:**
    - Clone the repository locally
    - Run `python data/mock_data_generator.py`
    - Run `python models/train_model.py`
    - Start FastAPI: `uvicorn api.app:app --reload`
    - Start Dashboard: `streamlit run dashboard/app.py`
    """)
    
    # Auto-refresh
    if auto_refresh:
        time.sleep(refresh_rate)
        st.rerun()

if __name__ == "__main__":
    main()