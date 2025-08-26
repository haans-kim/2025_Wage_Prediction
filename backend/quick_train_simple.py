#!/usr/bin/env python3
"""
Quick simple training script - trains both models with Linear Regression only (fastest)
"""

import requests
import json
import time

BASE_URL = "http://localhost:8000"

def train_simple_models():
    """Train both models with simple Linear Regression for speed"""
    
    print("\n" + "=" * 70)
    print("🚀 간단한 듀얼 모델 훈련 (Linear Regression만 사용)")
    print("=" * 70)
    
    # 1. 데이터 로드
    print("\n1️⃣ 데이터 로딩...")
    response = requests.post(f"{BASE_URL}/api/data/load-master")
    if response.status_code != 200:
        print(f"❌ 데이터 로드 실패: {response.text}")
        return
    print("✅ 데이터 로드 완료")
    
    models = [
        ("Base-up", "wage_increase_bu_sbl"),
        ("성과급", "wage_increase_mi_sbl")
    ]
    
    total_start = time.time()
    results = {}
    
    for model_name, target_column in models:
        print(f"\n{'='*70}")
        print(f"📌 {model_name} 모델 훈련")
        print(f"   타겟 변수: {target_column}")
        print(f"{'='*70}")
        
        model_start = time.time()
        
        # 환경 설정
        print(f"  ⚙️  환경 설정 중...")
        response = requests.post(
            f"{BASE_URL}/api/modeling/setup",
            json={
                "target_column": target_column,
                "train_size": 0.8,
                "session_id": 42
            }
        )
        if response.status_code != 200:
            print(f"  ❌ 환경 설정 실패: {response.text}")
            continue
        print(f"  ✅ 환경 설정 완료 ({time.time() - model_start:.1f}초)")
        
        # Linear Regression으로 바로 훈련 (비교 없음)
        print(f"\n  🔨 Linear Regression 모델 훈련 중...")
        train_start = time.time()
        response = requests.post(
            f"{BASE_URL}/api/modeling/train",
            json={
                "model_name": "lr",
                "tune_hyperparameters": False  # 하이퍼파라미터 튜닝 안함 (속도 우선)
            }
        )
        
        if response.status_code != 200:
            print(f"  ❌ 모델 훈련 실패: {response.text}")
            continue
        
        train_elapsed = time.time() - train_start
        data = response.json()
        
        results[model_name] = {
            'model': 'lr',
            'metrics': data.get('metrics', {}),
            'target': target_column,
            'time': time.time() - model_start
        }
        
        print(f"  ✅ {model_name} 모델 훈련 완료! (훈련시간: {train_elapsed:.1f}초)")
        
        if data.get('metrics'):
            metrics = data['metrics']
            print(f"  📊 성능 지표:")
            if 'MAE' in metrics:
                print(f"     - MAE: {metrics['MAE']:.4f}")
            if 'R2' in metrics:
                print(f"     - R2 Score: {metrics['R2']:.4f}")
    
    # 최종 요약
    total_elapsed = time.time() - total_start
    print(f"\n{'='*70}")
    print("🎯 훈련 완료 요약")
    print(f"{'='*70}")
    
    for model_name, info in results.items():
        print(f"\n【{model_name}】")
        print(f"  ├─ 타겟: {info['target']}")
        print(f"  ├─ 알고리즘: Linear Regression")
        print(f"  ├─ 소요시간: {info['time']:.1f}초")
        if info.get('metrics'):
            mae = info['metrics'].get('MAE', 'N/A')
            r2 = info['metrics'].get('R2', 'N/A')
            if mae != 'N/A':
                print(f"  ├─ MAE: {mae:.4f}")
            if r2 != 'N/A':
                print(f"  └─ R2: {r2:.4f}")
    
    print(f"\n⏱️  전체 소요시간: {total_elapsed:.1f}초")
    print(f"\n✅ 두 모델 모두 훈련 완료!")
    print("💡 이제 Dashboard에서 Feature Importance 기반 시뮬레이션을 사용할 수 있습니다.")

if __name__ == "__main__":
    train_simple_models()