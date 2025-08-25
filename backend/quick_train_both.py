#!/usr/bin/env python3
"""
Quick script to train both Base-up and Performance models
"""

import requests
import json

BASE_URL = "http://localhost:8000"

def train_both_models():
    """Train both models automatically"""
    
    print("🚀 자동 듀얼 모델 훈련 시작...")
    print("=" * 50)
    
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
    
    results = {}
    
    for model_name, target_column in models:
        print(f"\n{'='*50}")
        print(f"🎯 {model_name} 모델 훈련 ({target_column})")
        print(f"{'='*50}")
        
        # 환경 설정
        print(f"  ⚙️ 환경 설정 중...")
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
        print(f"  ✅ 환경 설정 완료")
        
        # 모델 비교 (선택적)
        print(f"  📊 모델 비교 중...")
        response = requests.post(f"{BASE_URL}/api/modeling/compare?n_select=3")
        best_model = 'lr'  # 기본값
        if response.status_code == 200:
            data = response.json()
            if data.get('comparison_results'):
                best_model = data['comparison_results'][0]['Model']
                print(f"  ✅ 최적 모델: {best_model}")
            else:
                print(f"  ⚠️ 비교 결과 없음, 기본 모델(lr) 사용")
        
        # 모델 훈련
        print(f"  🧠 {best_model} 모델 훈련 중...")
        response = requests.post(
            f"{BASE_URL}/api/modeling/train",
            json={
                "model_name": best_model,
                "tune_hyperparameters": False
            }
        )
        if response.status_code != 200:
            print(f"  ❌ 모델 훈련 실패: {response.text}")
            continue
        
        data = response.json()
        results[model_name] = {
            'model': best_model,
            'metrics': data.get('metrics', {})
        }
        print(f"  ✅ {model_name} 모델 훈련 완료!")
        
        if 'metrics' in data:
            print(f"  📈 성능:")
            for metric, value in data['metrics'].items():
                print(f"     {metric}: {value}")
    
    # 최종 요약
    print(f"\n{'='*50}")
    print("✨ 훈련 완료!")
    print(f"{'='*50}")
    
    for model_name, info in results.items():
        print(f"\n{model_name} 모델:")
        print(f"  - 알고리즘: {info['model']}")
        if info['metrics']:
            print(f"  - R2 Score: {info['metrics'].get('R2', 'N/A')}")
            print(f"  - MAE: {info['metrics'].get('MAE', 'N/A')}")
    
    if len(results) == 2:
        print("\n🎉 두 모델 모두 준비되었습니다!")
        print("   Base-up + 성과급 = 전체 인상률")
        print("\n이제 Dashboard에서 예측 및 분석을 사용할 수 있습니다.")
    else:
        print(f"\n⚠️ {len(results)}/2 모델만 훈련되었습니다.")

if __name__ == "__main__":
    train_both_models()