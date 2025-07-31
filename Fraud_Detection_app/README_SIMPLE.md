# Simple Credit Card Fraud Detection

A **reliable, single-file** machine learning system for detecting fraudulent credit card transactions.

## Why This Version?

- **Simple**: Single Python file, easy to understand
- **Reliable**: No complex dependencies, works everywhere
- **Fast**: Quick training and prediction
- **Visual**: Beautiful charts and interactive dashboard
- **Production-ready**: Save/load models, make predictions

## Installation

### Option 1: Quick Install (Recommended)

```bash
pip install -r requirements_simple.txt
```

### Option 2: Manual Install

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib streamlit plotly
```

## Quick Start

### 1. Run the System

```bash
python simple_fraud_detection.py
```

### 2. Launch Dashboard

```bash
streamlit run simple_dashboard.py
```

## Files

- `simple_fraud_detection.py` - Main fraud detection system
- `simple_dashboard.py` - Interactive web dashboard
- `requirements_simple.txt` - Minimal dependencies
- `README_SIMPLE.md` - This file

## Features

### Core System (`simple_fraud_detection.py`)

- **Data Generation**: Realistic synthetic credit card data
- **Feature Engineering**: Time, amount, and statistical features
- **Model Training**: Logistic Regression + Random Forest
- **Evaluation**: ROC curves, confusion matrices, feature importance
- **Prediction**: Real-time fraud detection for new transactions
- **Persistence**: Save and load trained models

### Dashboard (`simple_dashboard.py`)

- **Overview**: System status and quick actions
- **Training**: Interactive model training with parameters
- **Prediction**: Real-time fraud prediction interface
- **Results**: Visual charts and performance metrics

## Example Usage

```python
from simple_fraud_detection import SimpleFraudDetector

# Create detector
detector = SimpleFraudDetector()

# Train model
results = detector.run_full_pipeline(n_samples=10000, fraud_ratio=0.01)

# Make prediction
transaction = {
    'Time': 36000,      # 10 AM
    'Amount': 150.0,    # $150
    'V1': 1.2,         # Feature V1
    'V2': -0.8,        # Feature V2
    'V3': 0.5,         # Feature V3
    'V4': -1.1,        # Feature V4
    'V5': 0.3          # Feature V5
}

prediction = detector.predict_new_transaction(transaction)
print(f"Fraud: {prediction['is_fraud']}")
print(f"Probability: {prediction['probability']:.3f}")
```

## Expected Performance

- **AUC Score**: 0.95+
- **Precision**: 0.90+
- **Recall**: 0.85+
- **Training Time**: < 30 seconds
- **Prediction Time**: < 1 second

## Customization

### Adjust Training Parameters

```python
# More data, different fraud ratio
results = detector.run_full_pipeline(
    n_samples=20000,    # More samples
    fraud_ratio=0.02    # Higher fraud rate
)
```

### Add Custom Features

```python
def engineer_features(self, df):
    df_eng = df.copy()
    
    # Your custom features here
    df_eng['custom_feature'] = df_eng['Amount'] * df_eng['V1']
    
    return df_eng
```

## Dashboard Features

### Training Page

- Adjust sample size and fraud ratio
- Real-time training progress
- Performance metrics display

### Prediction Page

- Interactive transaction input
- Quick preset buttons (Normal/Suspicious/High-risk)
- Real-time fraud probability

### Results Page

- ROC and Precision-Recall curves
- Confusion matrix visualization
- Feature importance charts

### Installation Issues

```bash
# If you get errors, try:
pip install --upgrade pip
pip install -r requirements_simple.txt --force-reinstall
```

### Dashboard Not Loading

```bash
# Check if streamlit is installed
pip install streamlit plotly

# Run with debug
streamlit run simple_dashboard.py --logger.level debug
```

### Model Training Fails

- Check Python version (3.7+ recommended)
- Ensure all dependencies are installed
- Try with smaller dataset first

## Output Files

After running the system, you'll get:

- `fraud_detection_results.png` - Performance charts
- `fraud_detection_model.joblib` - Trained model file

## Use Cases

- **Learning**: Understand fraud detection concepts
- **Prototyping**: Quick proof-of-concept
- **Teaching**: Educational tool for ML
- **Production**: Lightweight fraud detection system

## Future Enhancements

- Add more algorithms (SVM, Neural Networks)
- Real-time data streaming
- API endpoint for predictions
- Database integration
- Advanced feature engineering

## License

 For Education purpose - feel free to use and modify!

## Credits

Built with  using:

- **Scikit-learn**: Machine learning algorithms
- **Pandas**: Data manipulation
- **Matplotlib/Seaborn**: Visualizations
- **Streamlit**: Web dashboard
- **Plotly**: Interactive charts

---

**Ready to detect fraud? Run `python simple_fraud_detection.py` and get started!**
