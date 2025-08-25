#!/usr/bin/env python3
"""
Initialize the model with sample data for testing
"""

import asyncio
import requests
import json
from pathlib import Path

BASE_URL = "http://localhost:8000"

async def init_model():
    """Initialize model with sample data"""
    
    print("🚀 Starting model initialization...")
    
    # 1. Load master data
    print("\n1️⃣ Loading master data...")
    response = requests.post(f"{BASE_URL}/api/data/load-master")
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Master data loaded: {data['summary']['shape']} shape")
    else:
        print(f"❌ Failed to load master data: {response.text}")
        return
    
    # 2. Setup modeling environment
    print("\n2️⃣ Setting up modeling environment...")
    response = requests.post(
        f"{BASE_URL}/api/modeling/setup",
        json={
            "target_column": "wage_increase_total_sbl",
            "train_size": 0.8,
            "session_id": 42
        }
    )
    if response.status_code == 200:
        print("✅ Modeling environment setup complete")
    else:
        print(f"❌ Failed to setup modeling: {response.text}")
        return
    
    # 3. Compare models
    print("\n3️⃣ Comparing models...")
    response = requests.post(f"{BASE_URL}/api/modeling/compare?n_select=3")
    if response.status_code == 200:
        data = response.json()
        if data.get('comparison_results'):
            print("✅ Model comparison complete")
            print(f"   Best model: {data['comparison_results'][0]['Model']}")
            print(f"   R2 Score: {data['comparison_results'][0]['R2']:.4f}")
            best_model = data['comparison_results'][0]['Model']
        else:
            print("⚠️ No comparison results, using default model")
            best_model = 'lr'
    else:
        print(f"❌ Failed to compare models: {response.text}")
        best_model = 'lr'
    
    # 4. Train the best model
    print(f"\n4️⃣ Training {best_model} model...")
    response = requests.post(
        f"{BASE_URL}/api/modeling/train",
        json={
            "model_name": best_model,
            "tune_hyperparameters": False  # Skip tuning for speed
        }
    )
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Model trained successfully")
        if 'metrics' in data:
            print(f"   MAE: {data['metrics'].get('MAE', 'N/A')}")
            print(f"   R2: {data['metrics'].get('R2', 'N/A')}")
    else:
        print(f"❌ Failed to train model: {response.text}")
        return
    
    # 5. Evaluate model
    print("\n5️⃣ Evaluating model...")
    response = requests.get(f"{BASE_URL}/api/modeling/evaluate")
    if response.status_code == 200:
        print("✅ Model evaluation complete")
    else:
        print(f"⚠️ Model evaluation warning: {response.text}")
    
    # 6. Test prediction
    print("\n6️⃣ Testing prediction...")
    response = requests.post(f"{BASE_URL}/api/modeling/predict")
    if response.status_code == 200:
        data = response.json()
        if 'predictions' in data and len(data['predictions']) > 0:
            print(f"✅ Prediction test successful")
            pred = data['predictions'][0]
            if isinstance(pred, dict):
                # If prediction is a dict with label and prediction
                if 'Label' in pred:
                    print(f"   Sample prediction: {pred['Label']:.4f}")
                else:
                    print(f"   Prediction data: {pred}")
            else:
                print(f"   Sample prediction: {pred:.4f}")
        else:
            print("⚠️ Prediction returned but no results")
    else:
        print(f"⚠️ Prediction test warning: {response.text}")
    
    print("\n✨ Model initialization complete!")
    print("You can now use the Dashboard and Analysis pages.")

if __name__ == "__main__":
    asyncio.run(init_model())