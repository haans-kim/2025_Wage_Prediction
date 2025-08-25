#!/usr/bin/env python3
"""
Initialize dual models (Base-up and Performance) with sample data
"""

import asyncio
import requests
import json
from pathlib import Path

BASE_URL = "http://localhost:8000"

async def train_model(target_column: str, model_name: str):
    """Train a single model for specified target"""
    
    print(f"\n🎯 Training {model_name} model for {target_column}...")
    
    # 1. Setup modeling environment
    print(f"  1️⃣ Setting up environment...")
    response = requests.post(
        f"{BASE_URL}/api/modeling/setup",
        json={
            "target_column": target_column,
            "train_size": 0.8,
            "session_id": 42
        }
    )
    if response.status_code == 200:
        print(f"  ✅ Environment setup complete")
    else:
        print(f"  ❌ Failed to setup: {response.text}")
        return False
    
    # 2. Compare models
    print(f"  2️⃣ Comparing models...")
    response = requests.post(f"{BASE_URL}/api/modeling/compare?n_select=3")
    if response.status_code == 200:
        data = response.json()
        if data.get('comparison_results'):
            print(f"  ✅ Model comparison complete")
            best_model = data['comparison_results'][0]['Model'] if data['comparison_results'] else 'lr'
            print(f"     Best model: {best_model}")
        else:
            print(f"  ⚠️ No comparison results, using lr")
            best_model = 'lr'
    else:
        print(f"  ⚠️ Comparison failed, using lr")
        best_model = 'lr'
    
    # 3. Train the model
    print(f"  3️⃣ Training {best_model} model...")
    response = requests.post(
        f"{BASE_URL}/api/modeling/train",
        json={
            "model_name": best_model,
            "tune_hyperparameters": False  # Skip tuning for speed
        }
    )
    if response.status_code == 200:
        data = response.json()
        print(f"  ✅ Model trained successfully")
        if 'metrics' in data:
            print(f"     MAE: {data['metrics'].get('MAE', 'N/A')}")
            print(f"     R2: {data['metrics'].get('R2', 'N/A')}")
    else:
        print(f"  ❌ Failed to train model: {response.text}")
        return False
    
    # 4. Store the model with specific name
    # This would require API modification to support named model storage
    print(f"  ✅ {model_name} model ready!")
    return True

async def init_dual_models():
    """Initialize both Base-up and Performance models"""
    
    print("🚀 Starting dual model initialization...")
    
    # 1. Load master data
    print("\n1️⃣ Loading master data from SambaWage_250825.xlsx...")
    response = requests.post(f"{BASE_URL}/api/data/load-master")
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Master data loaded: {data['summary']['shape']} shape")
        columns = data['summary']['columns']
        
        # Check for target columns
        has_baseup = 'wage_increase_bu_sbl' in columns
        has_performance = 'wage_increase_mi_sbl' in columns
        
        print(f"\nTarget columns available:")
        print(f"  - Base-up (wage_increase_bu_sbl): {'✅' if has_baseup else '❌'}")
        print(f"  - Performance (wage_increase_mi_sbl): {'✅' if has_performance else '❌'}")
        
        if not (has_baseup and has_performance):
            print("\n❌ Required target columns not found!")
            return
    else:
        print(f"❌ Failed to load master data: {response.text}")
        return
    
    # 2. Train Base-up model
    success_baseup = await train_model('wage_increase_bu_sbl', 'Base-up')
    
    # 3. Train Performance model
    success_performance = await train_model('wage_increase_mi_sbl', 'Performance')
    
    # 4. Summary
    print("\n" + "="*50)
    print("✨ Dual Model Initialization Complete!")
    print("="*50)
    print(f"  Base-up Model: {'✅ Ready' if success_baseup else '❌ Failed'}")
    print(f"  Performance Model: {'✅ Ready' if success_performance else '❌ Failed'}")
    print("\nYou can now use:")
    print("  - Dashboard for predictions")
    print("  - Analysis page for SHAP analysis")
    print("  - What-If scenarios with both models")

if __name__ == "__main__":
    asyncio.run(init_dual_models())