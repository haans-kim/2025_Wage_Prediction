from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any, Optional, List
from pydantic import BaseModel
from app.services.modeling_service import modeling_service

router = APIRouter()

class ModelingSetupRequest(BaseModel):
    target_column: Optional[str] = None  # 자동 감지 가능
    train_size: Optional[float] = None
    session_id: int = 42  # 고정된 시드값 사용

class ModelTrainingRequest(BaseModel):
    model_name: str
    tune_hyperparameters: bool = True

@router.post("/setup")
async def setup_modeling(request: ModelingSetupRequest) -> Dict[str, Any]:
    """
    PyCaret 모델링 환경 설정
    """
    try:
        # PyCaret 사용 가능 여부 확인
        if not modeling_service.check_pycaret_availability():
            raise HTTPException(
                status_code=500, 
                detail="PyCaret is not installed. Please install it with: pip install pycaret"
            )
        
        # target_column이 없으면 자동 감지
        target_column = request.target_column
        if target_column is None:
            from app.services.data_service import data_service
            model_config = data_service.get_model_config()
            target_column = model_config.get('target_column')
            if target_column is None:
                # 마지막 컬럼 사용
                if data_service.current_data is not None:
                    target_column = data_service.current_data.columns[-1]
        
        # 환경 설정 실행
        result = modeling_service.setup_pycaret_environment(
            target_column=target_column,
            train_size=request.train_size,
            session_id=request.session_id
        )
        
        return {
            **result,
            "setup_request": {
                "target_column": target_column,
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
    try:
        result = modeling_service.compare_models_adaptive(n_select=n_select)
        
        # 모델 비교 완료 후 ExplainerDashboard 캐시 클리어
        try:
            from app.services.explainer_dashboard_service import explainer_dashboard_service
            from app.services.analysis_service import analysis_service
            
            # 기존 ExplainerDashboard 중지 (새로운 모델로 재생성 필요)
            if explainer_dashboard_service.is_running:
                explainer_dashboard_service.stop_dashboard()
                print("🔄 Stopped ExplainerDashboard for model comparison update")
            
            # Feature importance 캐시 클리어
            analysis_service._importance_cache.clear()
            analysis_service._shap_cache.clear()
            print("🧹 Cleared analysis caches after model comparison")
            
        except Exception as dashboard_error:
            print(f"⚠️ ExplainerDashboard update failed: {dashboard_error}")
        
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
        
        # 모델 학습 완료 후 ExplainerDashboard 재생성
        try:
            from app.services.explainer_dashboard_service import explainer_dashboard_service
            
            # 기존 ExplainerDashboard 중지
            if explainer_dashboard_service.is_running:
                explainer_dashboard_service.stop_dashboard()
                print("🔄 Stopped existing ExplainerDashboard for model update")
            
            # Feature importance 캐시 클리어 (새로운 모델 반영)
            from app.services.analysis_service import analysis_service
            analysis_service._importance_cache.clear()
            analysis_service._shap_cache.clear()
            print("🧹 Cleared analysis caches for new model")
            
            print("✅ ExplainerDashboard will be recreated on next request with new model data")
            
        except Exception as dashboard_error:
            print(f"⚠️ ExplainerDashboard update failed: {dashboard_error}")
            # Dashboard 오류가 있어도 모델 학습 결과는 반환
        
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