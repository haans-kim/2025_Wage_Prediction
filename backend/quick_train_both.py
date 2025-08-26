#!/usr/bin/env python3
"""
Quick script to train both Base-up and Performance models
"""

import requests
import json
import time
import sys

BASE_URL = "http://localhost:8000"

def train_both_models():
    """Train both models automatically"""
    
    print("\n" + "═" * 80)
    print(" " * 20 + "DUAL MODEL TRAINING SYSTEM")
    print("═" * 80)
    
    # 1. Data loading
    print("\n[STEP 1] Data Loading")
    print("  Loading master_data.pkl...")
    response = requests.post(f"{BASE_URL}/api/data/load-master")
    if response.status_code != 200:
        print(f"  ERROR: Data load failed - {response.text}")
        return
    data = response.json()
    if data.get('summary'):
        shape = data['summary'].get('shape', 'unknown')
        print(f"  SUCCESS: Data loaded - Shape: {shape}")
    
    models = [
        ("Base-up", "wage_increase_bu_sbl"),
        ("Performance", "wage_increase_mi_sbl")
    ]
    
    results = {}
    
    for model_name, target_column in models:
        print(f"\n" + "─" * 80)
        print(f"[MODEL] {model_name}")
        print(f"  Target Variable: {target_column}")
        print("─" * 80)
        
        # Environment configuration
        print(f"\n  [Configuration]")
        print(f"    - Train/Test Split: 80/20")
        print(f"    - Preprocessing: Normalization, Missing Value Imputation")
        print(f"    - Setting up environment...", end="")
        sys.stdout.flush()
        
        setup_start = time.time()
        response = requests.post(
            f"{BASE_URL}/api/modeling/setup",
            json={
                "target_column": target_column,
                "train_size": 0.8,
                "session_id": 42
            }
        )
        setup_elapsed = time.time() - setup_start
        
        if response.status_code != 200:
            print(f"\n    ERROR: Setup failed - {response.text}")
            continue
        print(f" Done! ({setup_elapsed:.1f}s)")
        
        # Model comparison
        print(f"\n  [Model Comparison]")
        print(f"    Comparing 5 models (1-3 minutes)...")
        print(f"    Models: Linear Regression, Ridge, Lasso, ElasticNet, Decision Tree")
        sys.stdout.flush()  # Immediate output
        
        start_time = time.time()
        response = requests.post(f"{BASE_URL}/api/modeling/compare?n_select=5")
        elapsed = time.time() - start_time
        
        print(f"    Comparison completed in {elapsed:.1f} seconds")
        best_model = 'lr'  # Default value
        best_metrics = {}
        
        if response.status_code == 200:
            data = response.json()
            if data.get('comparison_results'):
                print(f"\n  Model Comparison Results:")
                print(f"  " + "─" * 78)
                print(f"  │ {'Rank':<6} │ {'Model':<15} │ {'MAE':<12} │ {'RMSE':<12} │ {'R2 Score':<12} │")
                print(f"  " + "─" * 78)
                
                for i, result in enumerate(data['comparison_results'][:5], 1):
                    mae = result.get('MAE', 'N/A')
                    rmse = result.get('RMSE', 'N/A')
                    r2 = result.get('R2', 'N/A')
                    
                    # Formatting
                    mae_str = f"{mae:.4f}" if isinstance(mae, (int, float)) else str(mae)
                    rmse_str = f"{rmse:.4f}" if isinstance(rmse, (int, float)) else str(rmse)
                    r2_str = f"{r2:.4f}" if isinstance(r2, (int, float)) else str(r2)
                    
                    model_name = result['Model'].upper()
                    print(f"  │ {i:<6} │ {model_name:<15} │ {mae_str:<12} │ {rmse_str:<12} │ {r2_str:<12} │")
                    
                    # Store best model metrics
                    if i == 1:
                        best_metrics = {
                            'MAE': mae,
                            'RMSE': rmse,
                            'R2': r2
                        }
                
                print(f"  " + "─" * 78)
                best_model = data['comparison_results'][0]['Model']
                
                print(f"\n  Selected Best Model: {best_model.upper()}")
                print(f"     - MAE: {best_metrics.get('MAE', 'N/A')}")
                print(f"     - RMSE: {best_metrics.get('RMSE', 'N/A')}")
                print(f"     - R2 Score: {best_metrics.get('R2', 'N/A')}")
            else:
                print(f"    WARNING: No comparison results, using default model (lr)")
        
        # Model training
        print(f"\n  [Model Training]")
        print(f"    Algorithm: {best_model.upper()}")
        print(f"    Training model...", end="")
        
        train_start = time.time()
        response = requests.post(
            f"{BASE_URL}/api/modeling/train",
            json={
                "model_name": best_model,
                "tune_hyperparameters": False
            }
        )
        train_elapsed = time.time() - train_start
        print(f" Done! ({train_elapsed:.1f}s)")
        if response.status_code != 200:
            print(f"    ERROR: Training failed - {response.text}")
            continue
        
        data = response.json()
        model_type = 'Base-up' if 'bu' in target_column else 'Performance'
        results[model_type] = {
            'model': best_model,
            'metrics': best_metrics if best_metrics else data.get('metrics', {}),
            'target': target_column
        }
        
        print(f"\n  [Training Complete: {model_type}]")
        print(f"    Final Performance:")
        print(f"      Algorithm: {best_model.upper()}")
        print(f"      MAE:       {best_metrics.get('MAE', 'N/A')}")
        print(f"      RMSE:      {best_metrics.get('RMSE', 'N/A')}")
        print(f"      R2 Score:  {best_metrics.get('R2', 'N/A')}")
    
    # Final summary
    print("\n" + "═" * 80)
    print(" " * 25 + "TRAINING SUMMARY")
    print("═" * 80)
    
    if len(results) > 0:
        print("\n" + "─" * 80)
        print(f"│ {'Model':<15} │ {'Target Variable':<25} │ {'Algorithm':<10} │ {'R2 Score':<10} │")
        print("─" * 80)
        
        for model_type, info in results.items():
            target = info.get('target', 'N/A')[:25]
            algorithm = info['model'].upper()[:10]
            
            if info.get('metrics'):
                r2 = info['metrics'].get('R2', 'N/A')
                r2_str = f"{r2:.4f}" if isinstance(r2, (int, float)) else str(r2)
            else:
                r2_str = 'N/A'
                
            print(f"│ {model_type:<15} │ {target:<25} │ {algorithm:<10} │ {r2_str:<10} │")
        
        print("─" * 80)
    
    if len(results) == 2:
        print("\n" + "─" * 80)
        print("STATUS: SUCCESS - Both models trained successfully")
        print("─" * 80)
        
        # 모델을 파일로 저장 (API 호출)
        print("\n" + "─" * 80)
        print("SAVING MODELS TO DISK")
        print("─" * 80)
        response = requests.post(f"{BASE_URL}/api/modeling/save-models")
        if response.status_code == 200:
            save_result = response.json()
            if 'error' not in save_result:
                print("✅ Models saved successfully to 'saved_models' directory")
                print("   - baseup_model.pkl")
                print("   - performance_model.pkl")
                print("   - current_model.pkl")
            else:
                print(f"⚠️ Warning: {save_result.get('error', 'Unknown error')}")
        else:
            print(f"⚠️ Warning: Failed to save models - {response.text}")
        
        print("\nUSAGE:")
        print("  1. Base-up Model:      Target = wage_increase_bu_sbl")
        print("  2. Performance Model:  Target = wage_increase_mi_sbl")
        print("  3. Total Increase:     Base-up + Performance")
        print("\nNOTE: Feature importance data is now available for Dashboard simulations")
        print("\n📁 Models are automatically loaded when server starts")
    else:
        print(f"\nWARNING: Only {len(results)}/2 models were trained successfully")

if __name__ == "__main__":
    train_both_models()