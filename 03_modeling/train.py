import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_auc_score
import joblib
import os
import gc

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

DATA_DIR = PROJECT_ROOT
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, 'artifacts')

def train_model():
    data_path = os.path.join(DATA_DIR, 'processed_data.parquet')
    if not os.path.exists(data_path):
        print("Processed data not found. Please run feature engineering first.")
        return

    print("Loading processed data...")
    df = pd.read_parquet(data_path)
    
    # Separate target and features
    if 'TARGET' not in df.columns:
        print("TARGET column not found.")
        return

    # Filter out test data (where TARGET might be null if combined, but usually we just use train set)
    # Assuming processed_data contains training data with TARGET
    train_df = df[df['TARGET'].notnull()]
    
    y = train_df['TARGET']
    X = train_df.drop(columns=['TARGET', 'SK_ID_CURR'])

    # Clean feature names for LightGBM
    import re
    X.columns = ["".join (c if c.isalnum() else "_" for c in str(x)) for x in X.columns]
    
    # Save feature names for API
    feature_names = X.columns.tolist()
    joblib.dump(feature_names, os.path.join(ARTIFACTS_DIR, 'features.joblib'))
    
    # Clean up
    del df, train_df
    gc.collect()

    print(f"Training shape: {X.shape}")
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # LightGBM parameters
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'is_unbalance': True, # Handle imbalance
        'boosting_type': 'gbdt',
        'learning_rate': 0.02,
        'num_leaves': 31,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1
    }
    
    print("Training LightGBM model...")
    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
    
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        valid_sets=[dtrain, dval],
        callbacks=[lgb.early_stopping(stopping_rounds=100), lgb.log_evaluation(100)]
    )
    
    # Evaluate
    y_pred = model.predict(X_val, num_iteration=model.best_iteration)
    auc = roc_auc_score(y_val, y_pred)
    print(f"Validation AUC: {auc:.4f}")
    
    # Save model
    model_path = os.path.join(ARTIFACTS_DIR, 'lgbm_model.pkl')
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_model()
