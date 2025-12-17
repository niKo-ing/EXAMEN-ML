import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve, classification_report
import joblib
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, 'artifacts')
DATA_DIR = PROJECT_ROOT

def evaluate_model():
    print("Loading artifacts...")
    model_path = os.path.join(ARTIFACTS_DIR, 'lgbm_model.pkl')
    if not os.path.exists(model_path):
        print("Model not found. Please train the model first.")
        return

    model = joblib.load(model_path)
    
    # Load data (using processed data for simplicity, ideally should use a separate test set)
    # In a real scenario, we would have saved X_val and y_val separately.
    # For this demo, we'll reload processed data and split again (ensuring same seed)
    data_path = os.path.join(DATA_DIR, 'processed_data.parquet')
    if not os.path.exists(data_path):
        print("Data not found.")
        return

    print("Loading data for evaluation...")
    df = pd.read_parquet(data_path)
    train_df = df[df['TARGET'].notnull()]
    
    y = train_df['TARGET']
    X = train_df.drop(columns=['TARGET', 'SK_ID_CURR'])
    
    # Clean feature names to match training
    X.columns = ["".join (c if c.isalnum() else "_" for c in str(x)) for x in X.columns]

    # Split (must match train.py)
    from sklearn.model_selection import train_test_split
    _, X_val, _, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Predicting...")
    y_pred_prob = model.predict(X_val)
    y_pred_class = (y_pred_prob > 0.5).astype(int) # Default threshold
    
    # Metrics
    auc = roc_auc_score(y_val, y_pred_prob)
    print(f"AUC: {auc:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred_class))
    
    # Plots
    print("Generating plots...")
    os.makedirs(os.path.join(ARTIFACTS_DIR, 'plots'), exist_ok=True)
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_val, y_pred_class)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(ARTIFACTS_DIR, 'plots', 'confusion_matrix.png'))
    plt.close()
    
    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_val, y_pred_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig(os.path.join(ARTIFACTS_DIR, 'plots', 'roc_curve.png'))
    plt.close()
    
    # 3. Feature Importance
    plt.figure(figsize=(10, 8))
    lgb.plot_importance(model, max_num_features=20, importance_type='split')
    plt.title('Feature Importance (Split)')
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACTS_DIR, 'plots', 'feature_importance.png'))
    plt.close()
    
    print(f"Evaluation complete. Plots saved to {os.path.join(ARTIFACTS_DIR, 'plots')}")

if __name__ == "__main__":
    evaluate_model()
