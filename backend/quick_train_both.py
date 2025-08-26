#!/usr/bin/env python3
"""
Quick script to train both Base-up and Performance models
"""

import requests
import json

BASE_URL = "http://localhost:8000"

def train_both_models():
    """Train both models automatically"""
    
    print("\n" + "=" * 70)
    print("🚀 자동 듀얼 모델 훈련 시작...")
    print("=" * 70)
    
    # 1. 데이터 로드
    print("\n1️⃣ 데이터 로딩...")
    print("   📂 master_data.pkl 파일 로드 중...")
    response = requests.post(f"{BASE_URL}/api/data/load-master")
    if response.status_code != 200:
        print(f"❌ 데이터 로드 실패: {response.text}")
        return
    data = response.json()
    if data.get('summary'):
        shape = data['summary'].get('shape', 'unknown')
        print(f"✅ 데이터 로드 완료 - {shape}")
    
    models = [
        ("Base-up", "wage_increase_bu_sbl"),
        ("성과급", "wage_increase_mi_sbl")
    ]
    
    results = {}
    
    for model_name, target_column in models:
        print(f"\n{'='*70}")
        print(f"📌 {model_name} 모델 훈련")
        print(f"   타겟 변수: {target_column}")
        print(f"{'='*70}")
        
        # 환경 설정
        print(f"  ⚙️  PyCaret 환경 설정 중...")
        print(f"     - Train/Test 분할: 80/20")
        print(f"     - 전처리: 정규화, 결측치 처리")
        print(f"  ⏱️  환경 설정 진행 중 (약 10-20초 소요)...")
        
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
            print(f"  ❌ 환경 설정 실패: {response.text}")
            continue
        print(f"  ✅ 환경 설정 완료! (소요시간: {setup_elapsed:.1f}초)")
        
        # 모델 비교
        print(f"\n  📊 모델 비교 시작...")
        print(f"  ⏱️  5개 모델 비교 중 (약 1-3분 소요)...")
        print(f"  🔄 진행 중: Linear Regression, Ridge, Lasso, ElasticNet, Decision Tree...")
        
        import time
        start_time = time.time()
        response = requests.post(f"{BASE_URL}/api/modeling/compare?n_select=5")
        elapsed = time.time() - start_time
        
        print(f"  ⏱️  모델 비교 완료! (소요시간: {elapsed:.1f}초)")
        best_model = 'lr'  # 기본값
        best_metrics = {}
        
        if response.status_code == 200:
            data = response.json()
            if data.get('comparison_results'):
                print(f"\n  📈 모델 비교 결과:")
                print(f"  {'='*65}")
                print(f"  {'순위':<4} {'모델':<12} {'MAE':<10} {'RMSE':<10} {'R2':<10} {'MAPE':<10}")
                print(f"  {'-'*65}")
                
                for i, result in enumerate(data['comparison_results'][:5], 1):
                    mae = result.get('MAE', 'N/A')
                    rmse = result.get('RMSE', 'N/A')
                    r2 = result.get('R2', 'N/A')
                    mape = result.get('MAPE', 'N/A')
                    
                    # 포맷팅
                    mae_str = f"{mae:.4f}" if isinstance(mae, (int, float)) else str(mae)
                    rmse_str = f"{rmse:.4f}" if isinstance(rmse, (int, float)) else str(rmse)
                    r2_str = f"{r2:.4f}" if isinstance(r2, (int, float)) else str(r2)
                    mape_str = f"{mape:.2f}%" if isinstance(mape, (int, float)) else str(mape)
                    
                    print(f"  {i:<4} {result['Model']:<12} {mae_str:<10} {rmse_str:<10} {r2_str:<10} {mape_str:<10}")
                    
                    # 최고 모델의 메트릭 저장
                    if i == 1:
                        best_metrics = {
                            'MAE': mae,
                            'RMSE': rmse,
                            'R2': r2,
                            'MAPE': mape
                        }
                
                print(f"  {'='*65}")
                best_model = data['comparison_results'][0]['Model']
                
                print(f"\n  🏆 선택된 최적 모델: {best_model.upper()}")
                print(f"     - MAE: {best_metrics.get('MAE', 'N/A')}")
                print(f"     - RMSE: {best_metrics.get('RMSE', 'N/A')}")
                print(f"     - R2 Score: {best_metrics.get('R2', 'N/A')}")
            else:
                print(f"  ⚠️ 비교 결과 없음, 기본 모델(lr) 사용")
        
        # 모델 훈련
        print(f"\n  🔨 최종 모델 훈련 시작...")
        print(f"     선택된 알고리즘: {best_model.upper()}")
        print(f"  ⏱️  모델 훈련 중 (약 30초-1분 소요)...")
        
        train_start = time.time()
        response = requests.post(
            f"{BASE_URL}/api/modeling/train",
            json={
                "model_name": best_model,
                "tune_hyperparameters": False
            }
        )
        train_elapsed = time.time() - train_start
        print(f"  ⏱️  훈련 완료! (소요시간: {train_elapsed:.1f}초)")
        if response.status_code != 200:
            print(f"  ❌ 모델 훈련 실패: {response.text}")
            continue
        
        data = response.json()
        results[model_name] = {
            'model': best_model,
            'metrics': best_metrics if best_metrics else data.get('metrics', {}),
            'target': target_column
        }
        
        print(f"\n  ✅ {model_name} 모델 훈련 완료!")
        print(f"  📊 최종 성능:")
        print(f"     - 알고리즘: {best_model.upper()}")
        print(f"     - MAE: {best_metrics.get('MAE', 'N/A')}")
        print(f"     - RMSE: {best_metrics.get('RMSE', 'N/A')}")
        print(f"     - R2 Score: {best_metrics.get('R2', 'N/A')}")
    
    # 최종 요약
    print(f"\n{'='*70}")
    print("🎯 최종 훈련 결과 요약")
    print(f"{'='*70}")
    
    if len(results) > 0:
        print("\n📊 훈련된 모델 정보:\n")
        for model_name, info in results.items():
            print(f"  【{model_name}】")
            print(f"    ├─ 타겟 변수: {info.get('target', 'N/A')}")
            print(f"    ├─ 선택된 알고리즘: {info['model'].upper()}")
            
            if info.get('metrics'):
                mae = info['metrics'].get('MAE', 'N/A')
                rmse = info['metrics'].get('RMSE', 'N/A')
                r2 = info['metrics'].get('R2', 'N/A')
                
                mae_str = f"{mae:.4f}" if isinstance(mae, (int, float)) else str(mae)
                rmse_str = f"{rmse:.4f}" if isinstance(rmse, (int, float)) else str(rmse)
                r2_str = f"{r2:.4f}" if isinstance(r2, (int, float)) else str(r2)
                
                print(f"    ├─ MAE: {mae_str}")
                print(f"    ├─ RMSE: {rmse_str}")
                print(f"    └─ R2 Score: {r2_str}")
            print()
    
    if len(results) == 2:
        print(f"{'='*70}")
        print("✅ 성공: 두 모델 모두 훈련 완료!")
        print(f"{'='*70}")
        print("\n📌 사용 방법:")
        print("   1. Base-up 인상률 예측: wage_increase_bu_sbl")
        print("   2. 성과급 인상률 예측: wage_increase_mi_sbl")
        print("   3. 전체 인상률 = Base-up + 성과급")
        print("\n💡 이제 Dashboard에서 예측 및 분석을 사용할 수 있습니다.")
    else:
        print(f"\n⚠️ 경고: {len(results)}/2 모델만 훈련되었습니다.")

if __name__ == "__main__":
    train_both_models()