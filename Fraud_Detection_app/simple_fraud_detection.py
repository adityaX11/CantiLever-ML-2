#!/usr/bin/env python3
"""
Simple Credit Card Fraud Detection System
A reliable, single-file solution for fraud detection
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    precision_recall_curve, roc_curve
)
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

class SimpleFraudDetector:
    """
    Simple and reliable credit card fraud detection system
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.models = {}
        self.best_model = None
        self.best_score = 0
        self.feature_names = None
        
    def generate_synthetic_data(self, n_samples=10000, fraud_ratio=0.01):
        """
        Generate realistic synthetic credit card data
        """
        np.random.seed(self.random_state)
        
        # Calculate sample sizes
        n_normal = int(n_samples * (1 - fraud_ratio))
        n_fraud = n_samples - n_normal
        
        # Generate normal transactions
        normal_data = {
            'Time': np.random.uniform(0, 86400, n_normal),  # 24 hours in seconds
            'Amount': np.random.exponential(100, n_normal),  # Exponential distribution
            'V1': np.random.normal(0, 1, n_normal),
            'V2': np.random.normal(0, 1, n_normal),
            'V3': np.random.normal(0, 1, n_normal),
            'V4': np.random.normal(0, 1, n_normal),
            'V5': np.random.normal(0, 1, n_normal),
            'Class': np.zeros(n_normal)
        }
        
        # Generate fraud transactions (different patterns)
        fraud_data = {
            'Time': np.random.uniform(0, 86400, n_fraud),
            'Amount': np.random.exponential(200, n_fraud),  # Higher amounts
            'V1': np.random.normal(2, 1, n_fraud),  # Different means
            'V2': np.random.normal(-1, 1, n_fraud),
            'V3': np.random.normal(1, 1, n_fraud),
            'V4': np.random.normal(-2, 1, n_fraud),
            'V5': np.random.normal(0.5, 1, n_fraud),
            'Class': np.ones(n_fraud)
        }
        
        # Combine data
        data = {}
        for key in normal_data:
            data[key] = np.concatenate([normal_data[key], fraud_data[key]])
        
        return pd.DataFrame(data)
    
    def engineer_features(self, df):
        """
        Simple but effective feature engineering
        """
        df_eng = df.copy()
        
        # Time-based features
        df_eng['hour'] = (df_eng['Time'] // 3600) % 24
        df_eng['is_night'] = ((df_eng['hour'] >= 22) | (df_eng['hour'] <= 6)).astype(int)
        df_eng['is_business_hours'] = ((df_eng['hour'] >= 9) & (df_eng['hour'] <= 17)).astype(int)
        
        # Amount-based features
        df_eng['amount_log'] = np.log1p(df_eng['Amount'])
        df_eng['amount_squared'] = df_eng['Amount'] ** 2
        df_eng['high_amount'] = (df_eng['Amount'] > df_eng['Amount'].quantile(0.95)).astype(int)
        
        # Statistical features
        feature_cols = ['V1', 'V2', 'V3', 'V4', 'V5']
        df_eng['mean_features'] = df_eng[feature_cols].mean(axis=1)
        df_eng['std_features'] = df_eng[feature_cols].std(axis=1)
        df_eng['max_features'] = df_eng[feature_cols].max(axis=1)
        df_eng['min_features'] = df_eng[feature_cols].min(axis=1)
        
        # Interaction features
        df_eng['v1_v2_ratio'] = df_eng['V1'] / (df_eng['V2'] + 1e-8)
        df_eng['v3_v4_product'] = df_eng['V3'] * df_eng['V4']
        
        return df_eng
    
    def prepare_data(self, df):
        """
        Prepare data for training
        """
        # Feature engineering
        df_eng = self.engineer_features(df)
        
        # Select features (exclude original time and target)
        exclude_cols = ['Time', 'Class']
        feature_cols = [col for col in df_eng.columns if col not in exclude_cols]
        
        X = df_eng[feature_cols]
        y = df_eng['Class']
        
        # Store feature names
        self.feature_names = feature_cols
        
        return X, y
    
    def train_models(self, X_train, y_train):
        """
        Train multiple models and find the best one
        """
        print("🤖 Training models...")
        
        # Define models
        models = {
            'Logistic Regression': LogisticRegression(
                random_state=self.random_state, 
                max_iter=1000,
                class_weight='balanced'
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state,
                class_weight='balanced'
            )
        }
        
        # Train and evaluate each model
        for name, model in models.items():
            print(f"  Training {name}...")
            
            # Train on full data first
            model.fit(X_train, y_train)
            
            # Cross-validation score (handle edge cases)
            try:
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
                mean_cv_score = cv_scores.mean()
                cv_std = cv_scores.std()
            except:
                # Fallback to accuracy if ROC AUC fails
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
                mean_cv_score = cv_scores.mean()
                cv_std = cv_scores.std()
            
            # Store model and score
            self.models[name] = {
                'model': model,
                'cv_score': mean_cv_score,
                'cv_std': cv_std
            }
            
            print(f"    CV Score: {mean_cv_score:.4f} (+/- {cv_std * 2:.4f})")
            
            # Update best model
            if mean_cv_score > self.best_score:
                self.best_score = mean_cv_score
                self.best_model = name
        
        print(f"🏆 Best model: {self.best_model} (Score: {self.best_score:.4f})")
        return self.models
    
    def evaluate_models(self, X_test, y_test):
        """
        Evaluate models on test set
        """
        print("📊 Evaluating models...")
        
        results = {}
        
        for name, model_info in self.models.items():
            model = model_info['model']
            
            # Predictions
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            
            # Metrics
            auc_score = roc_auc_score(y_test, y_proba)
            
            results[name] = {
                'predictions': y_pred,
                'probabilities': y_proba,
                'auc_score': auc_score,
                'classification_report': classification_report(y_test, y_pred),
                'confusion_matrix': confusion_matrix(y_test, y_pred)
            }
            
            print(f"  {name}:")
            print(f"    AUC Score: {auc_score:.4f}")
            print(f"    {results[name]['classification_report']}")
        
        return results
    
    def plot_results(self, X_test, y_test, results):
        """
        Create visualization plots
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Fraud Detection Results', fontsize=16, fontweight='bold')
        
        # ROC Curves
        ax1 = axes[0, 0]
        for name, result in results.items():
            fpr, tpr, _ = roc_curve(y_test, result['probabilities'])
            ax1.plot(fpr, tpr, label=f'{name} (AUC = {result["auc_score"]:.3f})')
        
        ax1.plot([0, 1], [0, 1], 'k--', label='Random')
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('ROC Curves')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Precision-Recall Curves
        ax2 = axes[0, 1]
        for name, result in results.items():
            precision, recall, _ = precision_recall_curve(y_test, result['probabilities'])
            ax2.plot(recall, precision, label=name)
        
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('Precision-Recall Curves')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Confusion Matrix (Best Model)
        ax3 = axes[1, 0]
        best_result = results[self.best_model]
        cm = best_result['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3)
        ax3.set_title(f'Confusion Matrix - {self.best_model}')
        ax3.set_xlabel('Predicted')
        ax3.set_ylabel('Actual')
        
        # Feature Importance (if Random Forest is best)
        ax4 = axes[1, 1]
        if self.best_model == 'Random Forest':
            best_model = self.models[self.best_model]['model']
            importances = best_model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            # Plot top 10 features
            top_n = min(10, len(self.feature_names))
            ax4.bar(range(top_n), importances[indices[:top_n]])
            ax4.set_xticks(range(top_n))
            ax4.set_xticklabels([self.feature_names[i] for i in indices[:top_n]], rotation=45, ha='right')
            ax4.set_title('Feature Importance (Top 10)')
            ax4.set_ylabel('Importance')
        else:
            ax4.text(0.5, 0.5, 'Feature importance\navailable for Random Forest only', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Feature Importance')
        
        plt.tight_layout()
        plt.savefig('fraud_detection_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def predict_new_transaction(self, transaction_data):
        """
        Predict fraud for a new transaction
        """
        if self.best_model is None:
            raise ValueError("No trained model available. Please train the model first.")
        
        # Prepare the transaction data
        if isinstance(transaction_data, dict):
            df = pd.DataFrame([transaction_data])
        else:
            df = transaction_data.copy()
        
        # Apply feature engineering
        df_eng = self.engineer_features(df)
        
        # Select features
        X = df_eng[self.feature_names]
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Get prediction
        model = self.models[self.best_model]['model']
        probability = model.predict_proba(X_scaled)[0, 1]
        prediction = 1 if probability > 0.5 else 0
        
        return {
            'prediction': prediction,
            'probability': probability,
            'is_fraud': bool(prediction),
            'confidence': max(probability, 1 - probability)
        }
    
    def save_model(self, filepath='fraud_detection_model.joblib'):
        """
        Save the trained model
        """
        if self.best_model is None:
            raise ValueError("No trained model available. Please train the model first.")
        
        model_data = {
            'best_model_name': self.best_model,
            'best_model': self.models[self.best_model]['model'],
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'best_score': self.best_score
        }
        
        joblib.dump(model_data, filepath)
        print(f"✅ Model saved to {filepath}")
    
    def load_model(self, filepath='fraud_detection_model.joblib'):
        """
        Load a trained model
        """
        model_data = joblib.load(filepath)
        
        self.best_model = model_data['best_model_name']
        self.models = {self.best_model: {'model': model_data['best_model']}}
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.best_score = model_data['best_score']
        
        print(f"✅ Model loaded from {filepath}")
        print(f"🏆 Best model: {self.best_model} (Score: {self.best_score:.4f})")
    
    def run_full_pipeline(self, n_samples=10000, fraud_ratio=0.01, save_model=True):
        """
        Run the complete fraud detection pipeline
        """
        print("🚨 Simple Credit Card Fraud Detection System")
        print("=" * 50)
        
        # Step 1: Generate data
        print("📊 Generating synthetic data...")
        data = self.generate_synthetic_data(n_samples, fraud_ratio)
        print(f"   Data shape: {data.shape}")
        print(f"   Fraud ratio: {data['Class'].mean():.3f}")
        
        # Step 2: Prepare data
        print("\n🔧 Preparing data...")
        X, y = self.prepare_data(data)
        print(f"   Features: {len(self.feature_names)}")
        
        # Step 3: Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        # Step 4: Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"   Train shape: {X_train.shape}")
        print(f"   Test shape: {X_test.shape}")
        
        # Step 5: Train models
        print("\n🤖 Training models...")
        self.train_models(X_train_scaled, y_train)
        
        # Step 6: Evaluate models
        print("\n📊 Evaluating models...")
        results = self.evaluate_models(X_test_scaled, y_test)
        
        # Step 7: Plot results
        print("\n📈 Creating visualizations...")
        self.plot_results(X_test_scaled, y_test, results)
        
        # Step 8: Save model
        if save_model:
            self.save_model()
        
        print("\n" + "=" * 50)
        print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        
        return {
            'data_shape': data.shape,
            'n_features': len(self.feature_names),
            'best_model': self.best_model,
            'best_score': self.best_score,
            'test_results': results
        }

def main():
    """
    Main function to run the fraud detection system
    """
    # Create detector
    detector = SimpleFraudDetector(random_state=42)
    
    # Run pipeline
    results = detector.run_full_pipeline(n_samples=10000, fraud_ratio=0.01)
    
    # Example prediction
    print("\n🔮 Example Prediction:")
    example_transaction = {
        'Time': 36000,  # 10 AM
        'Amount': 150.0,
        'V1': 1.2,
        'V2': -0.8,
        'V3': 0.5,
        'V4': -1.1,
        'V5': 0.3
    }
    
    prediction = detector.predict_new_transaction(example_transaction)
    print(f"   Transaction: {example_transaction}")
    print(f"   Prediction: {'FRAUD' if prediction['is_fraud'] else 'NORMAL'}")
    print(f"   Probability: {prediction['probability']:.3f}")
    print(f"   Confidence: {prediction['confidence']:.3f}")
    
    print("\n🎉 System ready for use!")
    print("📁 Check 'fraud_detection_results.png' for visualizations")
    print("💾 Model saved as 'fraud_detection_model.joblib'")

if __name__ == "__main__":
    main() 