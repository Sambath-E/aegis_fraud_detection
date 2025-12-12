# AEGIS - Real-time Transaction Fraud Detection System

## Project Overview
A machine learning system that detects fraudulent transactions in real-time using Isolation Forest algorithm.

## Features
- Real-time fraud prediction API
- Interactive Streamlit dashboard
- Mock data generation and streaming simulation
- FastAPI backend with automatic model loading

## Tech Stack
- **Backend**: FastAPI, Python 3.12
- **ML**: Scikit-learn, Isolation Forest
- **Dashboard**: Streamlit
- **Data**: Pandas, NumPy

## Project Structure

aegis_fraud_detection/
├── data/ # Mock data and generators
├── models/ # ML models and training scripts
├── api/ # FastAPI application
├── dashboard/ # Streamlit dashboard
├── tests/ # Test scripts
└── utils/ # Utility functions