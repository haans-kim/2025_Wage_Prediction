#!/usr/bin/env python3
"""
Script to retrain and save models with current library versions
to fix compatibility issues
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np
from datetime import datetime

# Add the app directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# PyCaret imports
from pycaret.regression import (
    setup, create_model, finalize_model, predict_model
)

def retrain_models():
    """Retrain both Base-up and Performance models"""
    
    print("\n" + "=" * 80)
    print("🔧 MODEL RETRAINING SCRIPT")
    print("=" * 80)
    print(f"Timestamp: {datetime.now()}")
    print("=" * 80 + "\n")
    
    # Check if data exists
    data_path = "data/current_data.pkl"
    if not os.path.exists(data_path):
        print("❌ No data found at data/current_data.pkl")
        print("   Please upload data first through the application.")
        return False
    
    # Load data
    print("📊 Loading data...")
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    # Handle both dict and DataFrame formats
    if isinstance(data, dict):
        # If it's a dict with a 'data' key
        if 'data' in data:
            df = data['data']  # This should already be a DataFrame
            if not isinstance(df, pd.DataFrame):
                df = pd.DataFrame(df)
        else:
            # Try to convert dict directly to DataFrame
            df = pd.DataFrame(data)
    else:
        df = data
    
    print(f"   Loaded {len(df)} rows, {len(df.columns)} columns")
    
    # Create saved_models directory if it doesn't exist
    os.makedirs("saved_models", exist_ok=True)
    
    # Backup existing models
    backup_dir = f"saved_models/backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if any(os.path.exists(f"saved_models/{f}") for f in ["baseup_model.pkl", "performance_model.pkl", "current_model.pkl"]):
        os.makedirs(backup_dir, exist_ok=True)
        print(f"\n📦 Backing up existing models to {backup_dir}")
        for model_file in ["baseup_model.pkl", "performance_model.pkl", "current_model.pkl"]:
            old_path = f"saved_models/{model_file}"
            if os.path.exists(old_path):
                new_path = os.path.join(backup_dir, model_file)
                os.rename(old_path, new_path)
                print(f"   Moved {model_file} to backup")
    
    # Define target columns - check what's actually available
    # First try the expected columns, then fall back to what's available
    possible_targets = {
        'baseup': ['wage_increase_bu_sbl', 'wage_increase_rate', 'wage_increase'],
        'performance': ['wage_increase_mi_sbl', 'wage_increase_rate', 'wage_increase']
    }
    
    # Find actual target columns in the data
    targets = {}
    for model_type, candidates in possible_targets.items():
        for col in candidates:
            if col in df.columns:
                targets[model_type] = col
                break
    
    # If we only have one wage column, use it for both models
    wage_columns = [col for col in df.columns if 'wage' in col.lower()]
    if len(wage_columns) == 1:
        single_target = wage_columns[0]
        targets = {
            'baseup': single_target,
            'performance': single_target
        }
        print(f"\n📌 Using single target column for both models: {single_target}")
    elif not targets:
        print("❌ No suitable target columns found in data")
        print(f"   Available columns: {list(df.columns)}")
        return False
    
    # Train models for each target
    for model_type, target_col in targets.items():
        print(f"\n{'='*60}")
        print(f"Training {model_type.upper()} Model")
        print(f"{'='*60}")
        
        # Check if target column exists
        if target_col not in df.columns:
            print(f"⚠️ Target column '{target_col}' not found in data")
            print(f"   Available columns: {list(df.columns)}")
            continue
        
        # Prepare data - remove rows with missing target values
        train_df = df.dropna(subset=[target_col]).copy()
        
        # Remove year columns and other non-predictive features
        exclude_cols = ['year', 'Year', 'YEAR', '년도', '연도'] + [col for col in df.columns if 'wage_increase' in col.lower() and col != target_col]
        for col in exclude_cols:
            if col in train_df.columns:
                train_df = train_df.drop(columns=[col])
                print(f"   Excluded column: {col}")
        
        print(f"\n📊 Training data: {len(train_df)} rows")
        
        if len(train_df) < 10:
            print(f"⚠️ Insufficient data for {model_type} model (only {len(train_df)} rows)")
            continue
        
        try:
            # Setup PyCaret environment with appropriate fold count
            print("\n🔧 Setting up PyCaret environment...")
            # Use fewer folds for cross-validation based on data size
            n_folds = min(3, len(train_df) - 2)  # At least 2 samples per fold
            
            exp = setup(
                train_df,
                target=target_col,
                train_size=0.8,
                fold=n_folds,  # Use appropriate number of folds
                session_id=123,
                verbose=False,
                html=False,
                normalize=True,
                transformation=False,
                remove_outliers=False,
                feature_selection=False
            )
            print(f"   Using {n_folds}-fold cross validation")
            
            # Create a simple model (Random Forest for stability)
            print("🌲 Creating Random Forest model...")
            model = create_model('rf', verbose=False)
            
            # Finalize the model (train on entire dataset)
            print("✨ Finalizing model...")
            final_model = finalize_model(model)
            
            # Extract feature importance
            feature_importance = []
            try:
                if hasattr(final_model, 'feature_importances_'):
                    # For tree-based models
                    importances = final_model.feature_importances_
                    feature_names = train_df.drop(columns=[target_col]).columns.tolist()
                    
                    for feat, imp in zip(feature_names, importances):
                        feature_importance.append({
                            'feature': feat,
                            'importance': float(imp)
                        })
                    
                    # Sort by importance
                    feature_importance.sort(key=lambda x: x['importance'], reverse=True)
                    
                    # Add rank
                    for i, item in enumerate(feature_importance):
                        item['rank'] = i + 1
                    
                    print(f"📈 Extracted feature importance for {len(feature_importance)} features")
            except Exception as e:
                print(f"⚠️ Could not extract feature importance: {e}")
            
            # Save the model
            model_path = f"saved_models/{model_type}_model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump({
                    'model': final_model,
                    'feature_importance': feature_importance,
                    'target': target_col,
                    'training_date': datetime.now().isoformat(),
                    'training_rows': len(train_df),
                    'model_type': 'RandomForestRegressor'
                }, f)
            
            print(f"✅ {model_type.capitalize()} model saved to {model_path}")
            
            # Also save as current model if it's baseup
            if model_type == 'baseup':
                current_path = "saved_models/current_model.pkl"
                with open(current_path, 'wb') as f:
                    pickle.dump({
                        'model': final_model,
                        'feature_importance': feature_importance,
                        'target': target_col,
                        'training_date': datetime.now().isoformat(),
                        'training_rows': len(train_df),
                        'model_type': 'RandomForestRegressor'
                    }, f)
                print(f"✅ Also saved as current model to {current_path}")
            
        except Exception as e:
            print(f"❌ Error training {model_type} model: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "=" * 80)
    print("✨ MODEL RETRAINING COMPLETE")
    print("=" * 80 + "\n")
    
    return True

if __name__ == "__main__":
    success = retrain_models()
    sys.exit(0 if success else 1)