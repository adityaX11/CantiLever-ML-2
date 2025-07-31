#!/usr/bin/env python3
"""
Simple Streamlit Dashboard for Credit Card Fraud Detection
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
import os
from simple_fraud_detection import SimpleFraudDetector

# Page configuration
st.set_page_config(
    page_title="Simple Fraud Detection",
    page_icon="🚨",
    layout="wide"
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
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .fraud-alert {
        background-color: #ffebee;
        color: #c62828;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #c62828;
    }
    .success-metric {
        background-color: #e8f5e8;
        color: #2e7d32;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2e7d32;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_or_create_detector():
    """Load existing model or create new detector"""
    model_path = 'fraud_detection_model.joblib'
    
    if os.path.exists(model_path):
        detector = SimpleFraudDetector()
        try:
            detector.load_model(model_path)
            return detector, True
        except:
            pass
    
    return SimpleFraudDetector(), False

def main():
    """Main dashboard function"""
    
    # Header
    st.markdown('<h1 class="main-header"> Simple Credit Card Fraud Detection</h1>', unsafe_allow_html=True)
    
    # Load detector
    detector, model_loaded = load_or_create_detector()
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Overview", "🤖 Train Model", "🔮 Predict", "📊 Results"]
    )
    
    if page == "🏠 Overview":
        show_overview(detector, model_loaded)
    elif page == "🤖 Train Model":
        show_training(detector)
    elif page == "🔮 Predict":
        show_prediction(detector)
    elif page == "📊 Results":
        show_results()

def show_overview(detector, model_loaded):
    """Show overview page"""
    st.header("🏠 System Overview")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card" style="color: black; font-weight: bold;">
            <h3 style="color: black; font-weight: bold;">Model Status</h3>
            <p style="color: black; font-weight: bold;"><strong>Status:</strong> {'✅ Trained' if model_loaded else '❌ Not Trained'}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if model_loaded:
            st.markdown(f"""
            <div class="metric-card" style="color: black; font-weight: bold;">
                <h3 style="color: black; font-weight: bold;">Best Model</h3>
                <p style="color: black; font-weight: bold;"><strong>Algorithm:</strong> {detector.best_model}</p>
                <p style="color: black; font-weight: bold;"><strong>Score:</strong> {detector.best_score:.4f}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="metric-card" style="color: black; font-weight: bold;">
                <h3 style="color: black; font-weight: bold;">Best Model</h3>
                <p style="color: black; font-weight: bold;">No model trained yet</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col3:
        if model_loaded:
            st.markdown(f"""
            <div class="metric-card" style="color: black; font-weight: bold;">
                <h3 style="color: black; font-weight: bold;">Features</h3>
                <p style="color: black; font-weight: bold;"><strong>Count:</strong> {len(detector.feature_names)}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="metric-card" style="color: black; font-weight: bold;">
                <h3 style="color: black; font-weight: bold;">Features</h3>
                <p style="color: black; font-weight: bold;">Not available</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # System description
    st.subheader("📋 About This System")
    st.write("""
    This is a **simple and reliable** credit card fraud detection system that:
    
    - 🎯 **Detects fraudulent transactions** with high accuracy
    - 🚀 **Easy to use** - single file implementation
    - 🔧 **Minimal dependencies** - only core ML libraries
    - 📊 **Visual results** - clear charts and metrics
    - 💾 **Model persistence** - save and load trained models
    
    The system uses **Logistic Regression** and **Random Forest** algorithms with 
    advanced feature engineering to detect fraud patterns.
    """)
    
    # Quick actions
    st.subheader("🚀 Quick Actions")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🤖 Train New Model", type="primary"):
            st.session_state.train_model = True
            st.rerun()
    
    with col2:
        if st.button("🔮 Make Prediction"):
            st.session_state.show_prediction = True
            st.rerun()

def show_training(detector):
    """Show model training page"""
    st.header("🤖 Train Model")
    
    # Training parameters
    st.subheader("📋 Training Parameters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        n_samples = st.slider("Number of samples", 1000, 50000, 10000)
    
    with col2:
        fraud_ratio = st.slider("Fraud ratio", 0.001, 0.1, 0.01, 0.001)
    
    with col3:
        random_state = st.number_input("Random state", value=42, min_value=1, max_value=1000)
    
    # Training button
    if st.button("🚀 Start Training", type="primary"):
        with st.spinner("Training models..."):
            # Create new detector with specified parameters
            detector = SimpleFraudDetector(random_state=random_state)
            
            # Run training
            results = detector.run_full_pipeline(
                n_samples=n_samples, 
                fraud_ratio=fraud_ratio,
                save_model=True
            )
            
            st.success("✅ Training completed successfully!")
            
            # Display results
            st.subheader("📊 Training Results")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Data Shape", f"{results['data_shape'][0]:,} × {results['data_shape'][1]}")
            
            with col2:
                st.metric("Features", results['n_features'])
            
            with col3:
                st.metric("Best Model", results['best_model'])
            
            with col4:
                st.metric("Best Score", f"{results['best_score']:.4f}")
            
            # Show test results
            st.subheader("📈 Test Performance")
            
            for model_name, result in results['test_results'].items():
                with st.expander(f"{model_name} Results"):
                    st.write(f"**AUC Score:** {result['auc_score']:.4f}")
                    st.text(result['classification_report'])
                    
                    # Confusion matrix
                    cm = result['confusion_matrix']
                    fig = px.imshow(
                        cm, 
                        text_auto=True, 
                        aspect="auto",
                        labels=dict(x="Predicted", y="Actual", color="Count"),
                        title=f"Confusion Matrix - {model_name}"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Reload detector
            st.rerun()

def show_prediction(detector):
    """Show prediction page"""
    st.header("🔮 Fraud Prediction")
    
    if not hasattr(detector, 'best_model') or detector.best_model is None:
        st.warning("⚠️ No trained model available. Please train a model first.")
        return
    
    st.subheader("📝 Enter Transaction Details")

    # --- Preset Buttons (OUTSIDE the form) ---
    st.subheader("🎯 Quick Presets")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Normal Transaction"):
            st.session_state['time'] = 36000
            st.session_state['amount'] = 50.0
            st.session_state['v1'] = 0.0
            st.session_state['v2'] = 0.0
            st.session_state['v3'] = 0.0
            st.session_state['v4'] = 0.0
            st.session_state['v5'] = 0.0
    with col2:
        if st.button("Suspicious Transaction"):
            st.session_state['time'] = 72000
            st.session_state['amount'] = 500.0
            st.session_state['v1'] = 2.0
            st.session_state['v2'] = -1.0
            st.session_state['v3'] = 1.0
            st.session_state['v4'] = -2.0
            st.session_state['v5'] = 0.5
    with col3:
        if st.button("High-Risk Transaction"):
            st.session_state['time'] = 30000
            st.session_state['amount'] = 2000.0
            st.session_state['v1'] = 3.0
            st.session_state['v2'] = -2.0
            st.session_state['v3'] = 2.0
            st.session_state['v4'] = -3.0
            st.session_state['v5'] = 1.0

    # Set default values if not already set
    if 'time' not in st.session_state:
        st.session_state['time'] = 36000
    if 'amount' not in st.session_state:
        st.session_state['amount'] = 100.0
    if 'v1' not in st.session_state:
        st.session_state['v1'] = 0.0
    if 'v2' not in st.session_state:
        st.session_state['v2'] = 0.0
    if 'v3' not in st.session_state:
        st.session_state['v3'] = 0.0
    if 'v4' not in st.session_state:
        st.session_state['v4'] = 0.0
    if 'v5' not in st.session_state:
        st.session_state['v5'] = 0.0

    # --- Input form ---
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            time = st.number_input("Time (seconds)", value=st.session_state['time'], help="Time of transaction in seconds (0-86400)")
            amount = st.number_input("Amount ($)", value=st.session_state['amount'], min_value=0.1, help="Transaction amount")
            v1 = st.number_input("V1", value=st.session_state['v1'], help="Feature V1")
            v2 = st.number_input("V2", value=st.session_state['v2'], help="Feature V2")
            v3 = st.number_input("V3", value=st.session_state['v3'], help="Feature V3")
        with col2:
            v4 = st.number_input("V4", value=st.session_state['v4'], help="Feature V4")
            v5 = st.number_input("V5", value=st.session_state['v5'], help="Feature V5")
        submitted = st.form_submit_button("🔮 Predict Fraud")

    if submitted:
        # Update session state with current values
        st.session_state['time'] = time
        st.session_state['amount'] = amount
        st.session_state['v1'] = v1
        st.session_state['v2'] = v2
        st.session_state['v3'] = v3
        st.session_state['v4'] = v4
        st.session_state['v5'] = v5

        # Create transaction data
        transaction = {
            'Time': time,
            'Amount': amount,
            'V1': v1,
            'V2': v2,
            'V3': v3,
            'V4': v4,
            'V5': v5
        }
        try:
            prediction = detector.predict_new_transaction(transaction)
            st.subheader("🎯 Prediction Results")
            if prediction['is_fraud']:
                st.markdown("""
                <div class="fraud-alert">
                    <h3>🚨 FRAUD DETECTED!</h3>
                    <p>This transaction has been flagged as potentially fraudulent.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="success-metric">
                    <h3>✅ NORMAL TRANSACTION</h3>
                    <p>This transaction appears to be legitimate.</p>
                </div>
                """, unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Fraud Probability", f"{prediction['probability']:.3f}")
            with col2:
                st.metric("Confidence", f"{prediction['confidence']:.3f}")
            with col3:
                st.metric("Prediction", "FRAUD" if prediction['is_fraud'] else "NORMAL")
            st.subheader("📋 Transaction Details")
            st.json(transaction)
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")

def show_results():
    """Show results page"""
    st.header("📊 Results & Visualizations")
    
    # Check if results file exists
    if os.path.exists('fraud_detection_results.png'):
        st.subheader("📈 Model Performance Charts")
        st.image('fraud_detection_results.png', use_column_width=True)
        
        st.markdown("""
        ### 📊 Chart Explanation
        
        **Top Left - ROC Curves**: Shows the trade-off between true positive rate and false positive rate.
        Higher AUC values indicate better performance.
        
        **Top Right - Precision-Recall Curves**: Shows the relationship between precision and recall.
        Important for imbalanced datasets like fraud detection.
        
        **Bottom Left - Confusion Matrix**: Shows actual vs predicted classifications.
        - True Negatives (top-left): Correctly identified normal transactions
        - False Positives (top-right): Normal transactions flagged as fraud
        - False Negatives (bottom-left): Fraud transactions missed
        - True Positives (bottom-right): Correctly identified fraud
        
        **Bottom Right - Feature Importance**: Shows which features are most important for predictions
        (only available for Random Forest model).
        """)
    else:
        st.info("📊 No results available yet. Train a model to see visualizations.")
    
    # Model file info
    if os.path.exists('fraud_detection_model.joblib'):
        st.subheader("💾 Saved Model")
        st.success("✅ Model file found: `fraud_detection_model.joblib`")
        
        # File info
        file_size = os.path.getsize('fraud_detection_model.joblib') / 1024  # KB
        st.metric("Model File Size", f"{file_size:.1f} KB")
    else:
        st.warning("⚠️ No saved model found. Train a model to save it.")

if __name__ == "__main__":
    main() 