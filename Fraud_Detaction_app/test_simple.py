#!/usr/bin/env python3
"""
Simple test script for the fraud detection system
"""

import sys
import os
import pandas as pd
import numpy as np
from simple_fraud_detection import SimpleFraudDetector

def test_data_generation():
    """Test data generation"""
    print("🧪 Testing data generation...")
    
    detector = SimpleFraudDetector()
    data = detector.generate_synthetic_data(n_samples=1000, fraud_ratio=0.01)
    
    assert data.shape[0] == 1000, f"Expected 1000 rows, got {data.shape[0]}"
    assert data.shape[1] == 8, f"Expected 8 columns, got {data.shape[1]}"
    assert 'Class' in data.columns, "Missing 'Class' column"
    assert data['Class'].mean() > 0, "No fraud cases generated"
    
    print("✅ Data generation test passed!")

def test_feature_engineering():
    """Test feature engineering"""
    print("🧪 Testing feature engineering...")
    
    detector = SimpleFraudDetector()
    data = detector.generate_synthetic_data(n_samples=100, fraud_ratio=0.01)
    df_eng = detector.engineer_features(data)
    
    # Check that new features were added
    original_cols = ['Time', 'Amount', 'V1', 'V2', 'V3', 'V4', 'V5', 'Class']
    new_features = [col for col in df_eng.columns if col not in original_cols]
    
    assert len(new_features) > 0, "No new features were created"
    assert 'hour' in df_eng.columns, "Missing 'hour' feature"
    assert 'amount_log' in df_eng.columns, "Missing 'amount_log' feature"
    
    print("✅ Feature engineering test passed!")

def test_model_training():
    """Test model training"""
    print("🧪 Testing model training...")
    
    detector = SimpleFraudDetector()
    data = detector.generate_synthetic_data(n_samples=1000, fraud_ratio=0.05)  # Higher fraud ratio
    X, y = detector.prepare_data(data)
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    X_train_scaled = detector.scaler.fit_transform(X_train)
    X_test_scaled = detector.scaler.transform(X_test)
    
    # Train models
    models = detector.train_models(X_train_scaled, y_train)
    
    assert len(models) > 0, "No models were trained"
    assert detector.best_model is not None, "No best model selected"
    assert detector.best_score > 0, "Best score should be positive"
    
    print("✅ Model training test passed!")

def test_prediction():
    """Test prediction functionality"""
    print("🧪 Testing prediction...")
    
    detector = SimpleFraudDetector()
    
    # Train a quick model
    data = detector.generate_synthetic_data(n_samples=1000, fraud_ratio=0.05)  # Higher fraud ratio
    X, y = detector.prepare_data(data)
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    X_train_scaled = detector.scaler.fit_transform(X_train)
    detector.train_models(X_train_scaled, y_train)
    
    # Test prediction
    transaction = {
        'Time': 36000,
        'Amount': 100.0,
        'V1': 0.0,
        'V2': 0.0,
        'V3': 0.0,
        'V4': 0.0,
        'V5': 0.0
    }
    
    prediction = detector.predict_new_transaction(transaction)
    
    assert 'prediction' in prediction, "Missing 'prediction' key"
    assert 'probability' in prediction, "Missing 'probability' key"
    assert 'is_fraud' in prediction, "Missing 'is_fraud' key"
    assert 'confidence' in prediction, "Missing 'confidence' key"
    assert 0 <= prediction['probability'] <= 1, "Probability should be between 0 and 1"
    
    print("✅ Prediction test passed!")

def test_model_persistence():
    """Test model save/load functionality"""
    print("🧪 Testing model persistence...")
    
    detector = SimpleFraudDetector()
    
    # Train a quick model
    data = detector.generate_synthetic_data(n_samples=1000, fraud_ratio=0.05)  # Higher fraud ratio
    X, y = detector.prepare_data(data)
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    X_train_scaled = detector.scaler.fit_transform(X_train)
    detector.train_models(X_train_scaled, y_train)
    
    # Save model
    test_file = 'test_model.joblib'
    detector.save_model(test_file)
    
    # Check file exists
    assert os.path.exists(test_file), "Model file was not created"
    
    # Load model
    new_detector = SimpleFraudDetector()
    new_detector.load_model(test_file)
    
    # Check that model was loaded correctly
    assert new_detector.best_model == detector.best_model, "Best model not loaded correctly"
    assert new_detector.best_score == detector.best_score, "Best score not loaded correctly"
    assert len(new_detector.feature_names) == len(detector.feature_names), "Feature names not loaded correctly"
    
    # Clean up
    os.remove(test_file)
    
    print("✅ Model persistence test passed!")

def run_all_tests():
    """Run all tests"""
    print("🚀 Running Simple Fraud Detection Tests")
    print("=" * 50)
    
    try:
        test_data_generation()
        test_feature_engineering()
        test_model_training()
        test_prediction()
        test_model_persistence()
        
        print("\n" + "=" * 50)
        print("🎉 ALL TESTS PASSED!")
        print("✅ System is working correctly")
        print("=" * 50)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        print("🔧 Please check your installation and try again")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1) 