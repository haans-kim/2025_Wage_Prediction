import pickle
import pandas as pd
from pycaret.regression import *
import warnings
warnings.filterwarnings('ignore')
import sys
import io

# stdout 억제
old_stdout = sys.stdout
sys.stdout = io.StringIO()

# 데이터 로드
with open('data/master_data.pkl', 'rb') as f:
    df = pickle.load(f)

# 2025년 데이터 제거
df = df[df['year'] != 2025].copy()

# Performance 모델 설정
setup(df, target='wage_increase_mi_sbl', session_id=42, verbose=False, train_size=0.8)

# Lasso 모델 생성
lasso_model = create_model('lasso', verbose=False)

# stdout 복원
sys.stdout = old_stdout

# 실제 모델 추출 (Pipeline에서)
if hasattr(lasso_model, 'steps'):
    actual_model = lasso_model.steps[-1][1]
    print(f'Pipeline에서 모델 추출: {type(actual_model).__name__}')
else:
    actual_model = lasso_model
    print(f'직접 모델 사용: {type(actual_model).__name__}')

# 계수 확인
if hasattr(actual_model, 'coef_'):
    coefs = actual_model.coef_
    X_train = get_config('X_train')
    feature_names = X_train.columns.tolist()
    
    print(f'\nLasso 모델의 계수 (총 {len(coefs)}개 features):')
    print('=' * 60)
    
    # 0이 아닌 계수
    non_zero = [(name, coef) for name, coef in zip(feature_names, coefs) if abs(coef) > 1e-10]
    
    if non_zero:
        print(f'\n✅ 0이 아닌 계수 ({len(non_zero)}개):')
        for name, coef in sorted(non_zero, key=lambda x: abs(x[1]), reverse=True):
            print(f'   {name:30s}: {coef:12.8f}')
    
    # 0에 가까운 계수 (매우 작지만 0은 아닌)
    small_non_zero = [(name, coef) for name, coef in zip(feature_names, coefs) 
                      if 1e-10 < abs(coef) < 1e-5]
    if small_non_zero:
        print(f'\n⚠️ 매우 작은 계수 ({len(small_non_zero)}개):')
        for name, coef in small_non_zero[:5]:  # 상위 5개만
            print(f'   {name:30s}: {coef:12.8f}')
    
    print(f'\n📊 요약:')
    print(f'   - 전체 features: {len(coefs)}개')
    print(f'   - 0이 아닌 features: {len(non_zero)}개') 
    print(f'   - 0인 features: {len(coefs) - len(non_zero)}개')
    
    # Feature importance 계산 (절대값 기준)
    importance = [(name, abs(coef)) for name, coef in zip(feature_names, coefs)]
    importance_sorted = sorted(importance, key=lambda x: x[1], reverse=True)
    
    print(f'\n🏆 Feature Importance (상위 10개):')
    total_importance = sum(imp for _, imp in importance)
    for i, (name, imp) in enumerate(importance_sorted[:10], 1):
        pct = (imp / total_importance * 100) if total_importance > 0 else 0
        print(f'   {i:2d}. {name:30s}: {pct:6.2f}%')
else:
    print('계수를 찾을 수 없습니다.')