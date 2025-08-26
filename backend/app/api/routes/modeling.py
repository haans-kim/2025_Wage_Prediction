from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any, Optional, List
from pydantic import BaseModel
from app.services.modeling_service import modeling_service

router = APIRouter()

class ModelingSetupRequest(BaseModel):
    target_column: str
    train_size: Optional[float] = None
    session_id: int = 123

class ModelTrainingRequest(BaseModel):
    model_name: str
    tune_hyperparameters: bool = True

class FeatureAdjustmentRequest(BaseModel):
    target: str  # 'baseup' or 'performance'
    feature_values: Dict[str, float]  # Feature name to adjusted value mapping
    use_baseline: bool = True  # Whether to use baseline values for non-adjusted features

@router.post("/setup")
async def setup_modeling(request: ModelingSetupRequest) -> Dict[str, Any]:
    """
    PyCaret 모델링 환경 설정
    """
    print(f"\n🔍 Setup Request: target={request.target_column}, train_size={request.train_size}, session_id={request.session_id}")
    try:
        # PyCaret 사용 가능 여부 확인
        if not modeling_service.check_pycaret_availability():
            raise HTTPException(
                status_code=500, 
                detail="PyCaret is not installed. Please install it with: pip install pycaret"
            )
        
        # 환경 설정 실행
        result = modeling_service.setup_pycaret_environment(
            target_column=request.target_column,
            train_size=request.train_size,
            session_id=request.session_id
        )
        
        return {
            **result,
            "setup_request": {
                "target_column": request.target_column,
                "train_size": request.train_size,
                "session_id": request.session_id
            }
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Modeling setup failed: {str(e)}")

@router.post("/compare")
async def compare_models(n_select: int = Query(default=3, ge=1, le=10)) -> Dict[str, Any]:
    """
    여러 ML 모델 비교 (데이터 크기에 적응적)
    """
    print(f"\n🔍 Compare Models: n_select={n_select}")
    try:
        result = modeling_service.compare_models_adaptive(n_select=n_select)
        
        # 상위 3개 모델 출력
        if 'comparison_results' in result and result['comparison_results']:
            print(f"🏆 Top 3 Models:")
            for i, model in enumerate(result['comparison_results'][:3], 1):
                print(f"   {i}. {model.get('Model', 'N/A')}: R2={model.get('R2', 'N/A')}")
        
        return {
            **result,
            "recommendation": "Use the recommended model for best performance, or choose from the best models list"
        }
        
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model comparison failed: {str(e)}")

@router.post("/train")
async def train_specific_model(request: ModelTrainingRequest) -> Dict[str, Any]:
    """
    특정 모델 학습 및 튜닝
    """
    try:
        result = modeling_service.train_specific_model(request.model_name)
        
        return {
            **result,
            "training_options": {
                "model_name": request.model_name,
                "hyperparameter_tuning": request.tune_hyperparameters
            }
        }
        
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model training failed: {str(e)}")

@router.get("/evaluate")
async def evaluate_current_model() -> Dict[str, Any]:
    """
    현재 학습된 모델의 성능 평가
    """
    try:
        result = modeling_service.get_model_evaluation()
        
        return result
        
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model evaluation failed: {str(e)}")

@router.post("/predict")
async def predict_with_model() -> Dict[str, Any]:
    """
    현재 모델로 테스트 데이터 예측
    """
    try:
        result = modeling_service.predict_with_model()
        
        return result
        
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@router.get("/available-models")
async def get_available_models() -> Dict[str, Any]:
    """
    현재 데이터에 사용 가능한 모델 목록 반환
    """
    try:
        from app.services.data_service import data_service
        
        if data_service.current_data is None:
            raise HTTPException(status_code=404, detail="No data loaded")
        
        data_size = len(data_service.current_data)
        optimal_settings = modeling_service.get_optimal_settings(data_size)
        
        model_descriptions = {
            'lr': 'Linear Regression - 선형 회귀',
            'ridge': 'Ridge Regression - 릿지 회귀',
            'lasso': 'Lasso Regression - 라쏘 회귀',
            'en': 'Elastic Net - 엘라스틱넷',
            'dt': 'Decision Tree - 의사결정트리',
            'rf': 'Random Forest - 랜덤포레스트',
            'gbr': 'Gradient Boosting - 그래디언트 부스팅',
            'xgboost': 'XGBoost - 익스트림 그래디언트 부스팅',
            'lightgbm': 'LightGBM - 라이트 그래디언트 부스팅'
        }
        
        available_models = []
        for model in optimal_settings['models']:
            available_models.append({
                'code': model,
                'name': model_descriptions.get(model, model),
                'recommended': model in optimal_settings['models'][:3]
            })
        
        return {
            'message': 'Available models retrieved successfully',
            'data_size': data_size,
            'data_size_category': 'small' if data_size < 30 else 'medium' if data_size < 100 else 'large',
            'available_models': available_models,
            'optimal_settings': {
                'train_size': optimal_settings['train_size'],
                'cv_folds': optimal_settings['cv_folds'],
                'preprocessing_enabled': {
                    'normalization': optimal_settings['normalize'],
                    'transformation': optimal_settings['transformation'],
                    'outlier_removal': optimal_settings['remove_outliers'],
                    'feature_selection': optimal_settings['feature_selection']
                }
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get available models: {str(e)}")

@router.get("/status")
async def get_modeling_status() -> Dict[str, Any]:
    """
    모델링 진행 상황 및 시스템 상태 확인
    """
    try:
        status = modeling_service.get_modeling_status()
        
        return {
            **status,
            "system_info": {
                "pycaret_installation_command": "pip install pycaret",
                "supported_tasks": ["regression"],
                "adaptive_modeling": True
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get modeling status: {str(e)}")

@router.delete("/clear")
async def clear_models() -> Dict[str, Any]:
    """
    모든 모델 및 실험 초기화
    """
    try:
        result = modeling_service.clear_models()
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear models: {str(e)}")

@router.get("/recommendations")
async def get_modeling_recommendations() -> Dict[str, Any]:
    """
    현재 데이터에 대한 모델링 권고사항
    """
    try:
        from app.services.data_service import data_service
        
        if data_service.current_data is None:
            raise HTTPException(status_code=404, detail="No data loaded")
        
        data_size = len(data_service.current_data)
        feature_count = len(data_service.current_data.columns) - 1  # target 제외
        
        recommendations = []
        
        if data_size < 30:
            recommendations.extend([
                "데이터가 적으므로 간단한 모델(선형회귀, 의사결정트리) 사용을 권장합니다.",
                "교차 검증 폴드 수를 3개로 제한합니다.",
                "과적합 방지를 위해 복잡한 전처리는 생략합니다."
            ])
        elif data_size < 100:
            recommendations.extend([
                "중간 크기 데이터로 앙상블 모델(랜덤포레스트, 그래디언트 부스팅) 사용 가능합니다.",
                "기본적인 전처리(정규화, 이상치 제거)를 적용합니다.",
                "특성 선택을 통해 모델 성능을 개선할 수 있습니다."
            ])
        else:
            recommendations.extend([
                "충분한 데이터로 고급 모델(XGBoost, LightGBM) 사용을 권장합니다.",
                "전체 전처리 파이프라인을 적용하여 최적 성능을 달성합니다.",
                "하이퍼파라미터 튜닝을 통해 성능을 극대화할 수 있습니다."
            ])
        
        if feature_count > data_size:
            recommendations.append("피처 수가 데이터 수보다 많으므로 특성 선택이 필수입니다.")
        
        return {
            'message': 'Modeling recommendations generated',
            'data_analysis': {
                'data_size': data_size,
                'feature_count': feature_count,
                'data_to_feature_ratio': round(data_size / feature_count, 2) if feature_count > 0 else 0
            },
            'recommendations': recommendations,
            'next_steps': [
                "1. 타겟 변수를 선택하여 모델링 환경을 설정하세요.",
                "2. 여러 모델을 비교하여 최적 모델을 찾으세요.",
                "3. 선택된 모델을 학습하고 성능을 평가하세요.",
                "4. 예측을 수행하고 결과를 분석하세요."
            ]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate recommendations: {str(e)}")

@router.get("/feature-importance/{target}")
async def get_feature_importance(
    target: str,
    top_n: int = Query(default=10, ge=1, le=50)
) -> Dict[str, Any]:
    """
    특정 모델의 feature importance 반환
    
    Args:
        target: 'baseup' or 'performance' or 'wage_increase_bu_sbl' or 'wage_increase_mi_sbl'
        top_n: 반환할 상위 feature 개수
    """
    try:
        # Feature importance 가져오기
        importance_list = modeling_service.get_feature_importance(target)
        
        if not importance_list:
            # Feature importance가 없는 경우, 모델이 학습되었는지 확인
            if target in ['baseup', 'wage_increase_bu_sbl']:
                if not modeling_service.baseup_model:
                    raise HTTPException(status_code=404, detail="Base-up model not trained yet")
            elif target in ['performance', 'wage_increase_mi_sbl']:
                if not modeling_service.performance_model:
                    raise HTTPException(status_code=404, detail="Performance model not trained yet")
            
            # 모델은 있지만 feature importance가 없는 경우
            raise HTTPException(status_code=404, detail="Feature importance not available. Please retrain the model.")
        
        # 상위 N개만 반환
        top_features = importance_list[:top_n] if len(importance_list) > top_n else importance_list
        
        # 현재 데이터에서 baseline 값 가져오기
        from app.services.data_service import data_service
        baseline_values = {}
        if data_service.current_data is not None:
            for feature_info in top_features:
                feature_name = feature_info['feature']
                if feature_name in data_service.current_data.columns:
                    # 최근 값 또는 평균값 사용
                    baseline_values[feature_name] = float(data_service.current_data[feature_name].iloc[-1])
        
        return {
            'message': 'Feature importance retrieved successfully',
            'target': target,
            'total_features': len(importance_list),
            'top_n': top_n,
            'features': top_features,
            'baseline_values': baseline_values
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get feature importance: {str(e)}")

@router.post("/predict-with-adjustments")
async def predict_with_adjustments(request: FeatureAdjustmentRequest) -> Dict[str, Any]:
    """
    조정된 feature 값으로 예측 수행
    """
    try:
        from app.services.data_service import data_service
        import pandas as pd
        
        # 모델 선택
        if request.target in ['baseup', 'wage_increase_bu_sbl']:
            model = modeling_service.baseup_model
            target_name = 'Base-up'
        elif request.target in ['performance', 'wage_increase_mi_sbl']:
            model = modeling_service.performance_model
            target_name = 'Performance'
        else:
            raise HTTPException(status_code=400, detail="Invalid target. Use 'baseup' or 'performance'")
        
        if not model:
            raise HTTPException(status_code=404, detail=f"{target_name} model not trained yet")
        
        # Baseline 데이터 준비 (최근 데이터 사용)
        if data_service.current_data is None:
            raise HTTPException(status_code=404, detail="No data loaded")
        
        # 최근 레코드를 baseline으로 사용
        baseline_data = data_service.current_data.iloc[-1:].copy()
        
        # Feature 값 조정
        for feature_name, adjusted_value in request.feature_values.items():
            if feature_name in baseline_data.columns:
                baseline_data[feature_name] = adjusted_value
            else:
                # Feature가 없는 경우 경고만 하고 계속 진행
                import warnings
                warnings.warn(f"Feature '{feature_name}' not found in data")
        
        # 예측 수행
        try:
            # PyCaret의 predict_model 사용
            from pycaret.regression import predict_model
            import sys
            import io
            
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            
            # 타겟 컬럼 제거 (있는 경우)
            target_cols = ['wage_increase_bu_sbl', 'wage_increase_mi_sbl']
            for col in target_cols:
                if col in baseline_data.columns:
                    baseline_data = baseline_data.drop(columns=[col])
            
            predictions = predict_model(model, data=baseline_data, verbose=False)
            sys.stdout = old_stdout
            
            # 예측 결과 추출
            if 'prediction_label' in predictions.columns:
                predicted_value = float(predictions['prediction_label'].iloc[0])
            elif 'Label' in predictions.columns:
                predicted_value = float(predictions['Label'].iloc[0])
            else:
                # 마지막 컬럼이 예측값일 가능성이 높음
                predicted_value = float(predictions.iloc[0, -1])
            
            # Baseline 예측 (조정 전)
            baseline_original = data_service.current_data.iloc[-1:].copy()
            for col in target_cols:
                if col in baseline_original.columns:
                    baseline_original = baseline_original.drop(columns=[col])
            
            sys.stdout = io.StringIO()
            baseline_predictions = predict_model(model, data=baseline_original, verbose=False)
            sys.stdout = old_stdout
            
            if 'prediction_label' in baseline_predictions.columns:
                baseline_value = float(baseline_predictions['prediction_label'].iloc[0])
            elif 'Label' in baseline_predictions.columns:
                baseline_value = float(baseline_predictions['Label'].iloc[0])
            else:
                baseline_value = float(baseline_predictions.iloc[0, -1])
            
            # 변화량 계산
            change = predicted_value - baseline_value
            change_percent = (change / baseline_value * 100) if baseline_value != 0 else 0
            
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {str(e)}")
        
        return {
            'message': 'Prediction with adjustments completed successfully',
            'target': request.target,
            'target_name': target_name,
            'baseline_prediction': baseline_value,
            'adjusted_prediction': predicted_value,
            'change': change,
            'change_percent': change_percent,
            'adjusted_features': request.feature_values,
            'feature_count': len(request.feature_values)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to predict with adjustments: {str(e)}")

@router.post("/save-models")
async def save_models() -> Dict[str, Any]:
    """
    학습된 모델을 파일로 저장
    """
    try:
        result = modeling_service.save_models()
        
        if 'error' in result:
            raise HTTPException(status_code=500, detail=result['error'])
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save models: {str(e)}")

@router.post("/load-models")
async def load_models() -> Dict[str, Any]:
    """
    저장된 모델을 파일에서 로드
    """
    try:
        result = modeling_service.load_saved_models()
        
        if 'error' in result:
            raise HTTPException(status_code=500, detail=result['error'])
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load models: {str(e)}")