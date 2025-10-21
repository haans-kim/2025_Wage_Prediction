import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import warnings
import io
import sys
from contextlib import redirect_stdout, redirect_stderr
import logging

# SHAP, LIME, scikit-learn imports with error handling
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logging.warning("SHAP not available. Install with: pip install shap")

try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    logging.warning("LIME not available. Install with: pip install lime")

try:
    from sklearn.inspection import permutation_importance, partial_dependence
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn not available for advanced analysis")

from app.services.data_service import data_service

class AnalysisService:
    def __init__(self):
        self.shap_explainer = None
        self.lime_explainer = None
        self.feature_names = None
        self.feature_display_names = {}  # 한글 표시명
        self.train_data = None
        self.test_data = None
        # 캐시 추가
        self._shap_cache = {}
        self._importance_cache = {}
        self._last_model_id = None
        
    def _get_training_data(self):
        """PyCaret 환경에서 학습 데이터 가져오기"""
        try:
            from pycaret.regression import get_config
            
            # PyCaret에서 데이터 가져오기
            X_train = get_config('X_train')
            y_train = get_config('y_train')
            X_test = get_config('X_test') 
            y_test = get_config('y_test')
            
            self.train_data = (X_train, y_train)
            self.test_data = (X_test, y_test)
            self.feature_names = list(X_train.columns)
            
            # 한글 컬럼명 매핑 가져오기
            self.feature_display_names = data_service.get_display_names(self.feature_names)
            
            return X_train, y_train, X_test, y_test
            
        except Exception as e:
            print(f"Warning: Could not get PyCaret data: {str(e)}")
            logging.warning(f"Could not get PyCaret data: {str(e)}")
            
            # PyCaret 환경이 필수임
            raise ValueError("PyCaret environment is required for analysis. Please run model training first.")
    
    def get_shap_analysis(self, model, sample_index: Optional[int] = None, top_n: int = 10) -> Dict[str, Any]:
        """SHAP 분석 수행"""
        
        if not SHAP_AVAILABLE:
            return {
                "error": "SHAP not available. Please install with: pip install shap",
                "available": False
            }
        
        # 캐시 키 생성
        model_id = id(model)
        cache_key = f"{model_id}_{top_n}"
        
        # 모델이 변경되었으면 캐시 초기화
        if self._last_model_id != model_id:
            self._shap_cache = {}
            self._importance_cache = {}
            self._last_model_id = model_id
        
        # 캐시에서 결과 확인
        if cache_key in self._shap_cache and sample_index is None:
            cached_result = self._shap_cache[cache_key]
            print(f"[DATA] Using cached SHAP analysis for model {model_id}")
            return cached_result
        
        try:
            # warnings 억제
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                X_train, y_train, X_test, y_test = self._get_training_data()
                
                if X_train is None:
                    raise ValueError("No training data available")
                
                # 데이터프레임을 numpy로 변환하고 모델의 feature names와 맞춤
                if hasattr(X_train, 'values'):
                    # 모델이 훈련된 feature names 가져오기
                    if hasattr(model, 'feature_names_in_'):
                        model_features = list(model.feature_names_in_)
                        print(f"[DATA] Model expects {len(model_features)} features: {model_features[:5]}...")
                        
                        # 모델이 기대하는 feature만 선택
                        available_features = [f for f in model_features if f in X_train.columns]
                        if len(available_features) == len(model_features):
                            X_train_filtered = X_train[model_features]
                            X_train_array = X_train_filtered.values
                            self.feature_names = model_features
                            print(f"[OK] Using {len(self.feature_names)} features matching model")
                        else:
                            print(f"[WARNING] Feature mismatch: available {len(available_features)}, needed {len(model_features)}")
                            # 기존 방식 사용
                            X_train_array = X_train.values
                            self.feature_names = X_train.columns.tolist()
                    else:
                        X_train_array = X_train.values
                        self.feature_names = X_train.columns.tolist()
                    
                    # 한글 컬럼명 매핑 가져오기
                    self.feature_display_names = data_service.get_display_names(self.feature_names)
                else:
                    X_train_array = X_train
                    self.feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
                
                # 데이터 정리 (NaN, inf 처리)
                X_train_array = np.nan_to_num(X_train_array, nan=0.0, posinf=1e6, neginf=-1e6)
                
                print(f"[DATA] SHAP Analysis: {len(self.feature_names)} features after preprocessing")
            
            # SHAP explainer 생성 (안전한 방식으로 수정)
            model_name = type(model).__name__.lower()
            
            # 데이터 준비 (모델이 기대하는 features와 일치하도록)
            if X_test is not None:
                if hasattr(model, 'feature_names_in_') and self.feature_names:
                    # X_test도 동일한 feature filtering 적용
                    if hasattr(X_test, 'columns'):
                        # DataFrame인 경우
                        available_test_features = [f for f in self.feature_names if f in X_test.columns]
                        if len(available_test_features) == len(self.feature_names):
                            X_test_filtered = X_test[self.feature_names]
                            analysis_data = X_test_filtered.values
                            print(f"[OK] X_test filtered to {len(self.feature_names)} features for SHAP")
                        else:
                            print(f"[WARNING] X_test missing some required features, using filtered X_train")
                            analysis_data = X_train_array[:100]
                    else:
                        # numpy array인 경우, feature 순서가 맞다고 가정
                        if X_test.shape[1] == len(self.feature_names):
                            analysis_data = X_test.copy()
                            print(f"[OK] X_test numpy array matches model features ({X_test.shape[1]})")
                        else:
                            print(f"[WARNING] X_test feature count mismatch ({X_test.shape[1]} vs {len(self.feature_names)}), using X_train")
                            analysis_data = X_train_array[:100]
                else:
                    analysis_data = X_test.values if hasattr(X_test, 'values') else X_test
            else:
                analysis_data = X_train_array[:100]
            
            analysis_data = analysis_data.copy()  # 복사본 생성
            
            # feature_names_in_ 속성 문제 방지
            try:
                # Tree-based models 시도
                if hasattr(model, 'feature_importances_'):
                    explainer = shap.TreeExplainer(model, feature_perturbation='tree_path_dependent')
                    shap_values = explainer.shap_values(analysis_data, check_additivity=False)
                else:
                    # 다른 모델들은 KernelExplainer 사용 (더 안전함)
                    n_background = min(20, len(X_train_array))  # 줄임
                    np.random.seed(42)
                    background_indices = np.random.choice(len(X_train_array), n_background, replace=False)
                    background_data = X_train_array[background_indices]
                    
                    explainer = shap.KernelExplainer(model.predict, background_data)
                    
                    n_samples = min(3, len(analysis_data))  # 크게 줄임
                    np.random.seed(42)
                    sample_indices = np.random.choice(len(analysis_data), n_samples, replace=False)
                    analysis_sample = analysis_data[sample_indices]
                    shap_values = explainer.shap_values(analysis_sample)
                    
            except Exception as e:
                print(f"[WARNING] SHAP TreeExplainer failed, using KernelExplainer: {e}")
                # 완전한 fallback - 모델을 래핑해서 feature_names_in_ 문제 해결
                try:
                    # 모델 예측 함수를 안전하게 래핑 (PyCaret용)
                    def safe_predict(X):
                        try:
                            # numpy 배열을 DataFrame으로 변환 (정확한 feature names 사용)
                            if hasattr(X, 'shape') and len(X.shape) == 2:
                                # 모델이 훈련된 정확한 feature names 사용
                                correct_feature_names = self.feature_names if self.feature_names else [f"feature_{i}" for i in range(X.shape[1])]
                                X_df = pd.DataFrame(X, columns=correct_feature_names)
                                predictions = model.predict(X_df)
                                print(f"[OK] SHAP predictions: shape={predictions.shape}, sample values={predictions[:3] if len(predictions) > 3 else predictions}")
                                return predictions
                            else:
                                # 1차원 배열인 경우
                                X_reshaped = X.reshape(1, -1) if len(X.shape) == 1 else X
                                correct_feature_names = self.feature_names if self.feature_names else [f"feature_{i}" for i in range(X_reshaped.shape[1])]
                                X_df = pd.DataFrame(X_reshaped, columns=correct_feature_names)
                                predictions = model.predict(X_df)
                                return predictions
                        except Exception as e:
                            print(f"[WARNING] SHAP safe_predict failed: {e}")
                            raise ValueError(f"Model prediction failed for SHAP analysis: {str(e)}")
                    
                    n_background = min(20, len(X_train_array))  # 줄임
                    np.random.seed(42)
                    background_indices = np.random.choice(len(X_train_array), n_background, replace=False)
                    background_data = X_train_array[background_indices]
                    
                    explainer = shap.KernelExplainer(safe_predict, background_data)
                    
                    n_samples = min(3, len(analysis_data))  # 크게 줄임
                    np.random.seed(42)
                    sample_indices = np.random.choice(len(analysis_data), n_samples, replace=False)
                    analysis_sample = analysis_data[sample_indices]
                    shap_values = explainer.shap_values(analysis_sample)
                    
                except Exception as inner_e:
                    print(f"[WARNING] KernelExplainer also failed: {inner_e}")
                    raise ValueError(f"All SHAP explainer methods failed. Model incompatible with SHAP analysis: {str(inner_e)}")
            
            # Feature importance 계산
            if isinstance(shap_values, np.ndarray):
                if len(shap_values.shape) > 1:
                    importance_scores = np.abs(shap_values).mean(axis=0)
                else:
                    importance_scores = np.abs(shap_values)
                
                print(f"[DATA] Raw importance scores: shape={importance_scores.shape}, values={importance_scores[:5]}")
                
                # 정규화를 통해 불균형 해결 (0-1 스케일로 정규화)
                if len(importance_scores) > 0:
                    # 이상치 제거를 위해 95 percentile로 클리핑
                    percentile_95 = np.percentile(importance_scores, 95)
                    importance_scores = np.clip(importance_scores, 0, percentile_95)
                    
                    # Min-Max 정규화 (0-1 스케일)
                    min_val = np.min(importance_scores)
                    max_val = np.max(importance_scores)
                    if max_val > min_val:
                        importance_scores = (importance_scores - min_val) / (max_val - min_val)
                    
                    # 퍼센트로 변환 (0-100%)
                    importance_scores = importance_scores * 100
                    
                    print(f"[DATA] Normalized importance scores: values={importance_scores[:5]}")
                
                print(f"[DATA] Final importance scores: shape={importance_scores.shape}, values={importance_scores[:5]}")
            else:
                importance_scores = np.abs(shap_values[0]).mean(axis=0) if len(shap_values) > 0 else []
                # 동일한 정규화 적용
                if len(importance_scores) > 0:
                    percentile_95 = np.percentile(importance_scores, 95)
                    importance_scores = np.clip(importance_scores, 0, percentile_95)
                    min_val = np.min(importance_scores)
                    max_val = np.max(importance_scores)
                    if max_val > min_val:
                        importance_scores = (importance_scores - min_val) / (max_val - min_val)
                    importance_scores = importance_scores * 100
            
            # 값이 모두 0인지 확인
            if np.all(importance_scores == 0):
                raise ValueError("Feature importance analysis failed - all scores are zero. Cannot provide meaningful analysis.")
            
            # Top N features
            feature_importance = []
            if len(importance_scores) > 0 and self.feature_names:
                for i, score in enumerate(importance_scores):
                    if i < len(self.feature_names):
                        english_name = self.feature_names[i]
                        korean_name = self.feature_display_names.get(english_name, english_name)
                        feature_importance.append({
                            "feature": english_name,
                            "feature_korean": korean_name,
                            "importance": float(score)
                        })
                
                # 중요도 순으로 정렬
                feature_importance.sort(key=lambda x: x["importance"], reverse=True)
                feature_importance = feature_importance[:top_n]
            
            # 개별 샘플 분석 (sample_index가 지정된 경우)
            sample_explanation = None
            if sample_index is not None and isinstance(shap_values, np.ndarray):
                if sample_index < len(shap_values):
                    sample_shap = shap_values[sample_index] if len(shap_values.shape) > 1 else shap_values
                    sample_explanation = {
                        "sample_index": sample_index,
                        "shap_values": sample_shap.tolist() if hasattr(sample_shap, 'tolist') else sample_shap,
                        "base_value": float(explainer.expected_value) if hasattr(explainer, 'expected_value') else 0
                    }
            
            result = {
                "message": "SHAP analysis completed successfully",
                "available": True,
                "feature_importance": feature_importance,
                "sample_explanation": sample_explanation,
                "explainer_type": type(explainer).__name__,
                "n_features": len(self.feature_names) if self.feature_names else 0,
                "n_samples_analyzed": len(shap_values) if isinstance(shap_values, np.ndarray) else 0
            }
            
            # 캐시에 저장 (sample_index가 없을 때만)
            if sample_index is None:
                self._shap_cache[cache_key] = result
                print(f"[DATA] Cached SHAP analysis for model {model_id}")
            
            return result
            
        except Exception as e:
            logging.error(f"SHAP analysis failed: {str(e)}")
            return {
                "error": f"SHAP analysis failed: {str(e)}",
                "available": False
            }
    
    def get_feature_importance(self, model, method: str = "shap", top_n: int = 15) -> Dict[str, Any]:
        """Feature importance 분석"""
        
        # 캐시 키 생성
        model_id = id(model)
        cache_key = f"{model_id}_{method}_{top_n}"
        
        # 캐시에서 결과 확인
        if cache_key in self._importance_cache:
            cached_result = self._importance_cache[cache_key]
            print(f"[DATA] Using cached {method} importance for model {model_id}")
            return cached_result
        
        try:
            X_train, y_train, X_test, y_test = self._get_training_data()
            
            if X_train is None:
                raise ValueError("No training data available")
            
            feature_importance = []
            
            if method == "shap" and SHAP_AVAILABLE:
                # SHAP 기반 feature importance - 동일한 분석 결과 사용
                shap_result = self.get_shap_analysis(model, top_n=top_n)
                if shap_result.get("available"):
                    feature_importance = shap_result.get("feature_importance", [])
                    # Feature importance 형식으로 재구성
                    result = {
                        "method": "shap",
                        "feature_importance": feature_importance
                    }
                    self._importance_cache[cache_key] = result
                    return result
            
            elif method == "pycaret":
                # PyCaret의 내장 해석 기능 사용
                try:
                    from pycaret.regression import get_config, interpret_model
                    
                    # PyCaret의 feature importance 가져오기
                    try:
                        # 변수 중요도 플롯 생성 (실제로는 플롯을 그리지 않고 데이터만 추출)
                        import matplotlib
                        matplotlib.use('Agg')  # 백엔드를 non-interactive로 설정
                        
                        # get_config로 feature importance 시도
                        feature_importance_df = get_config('feature_importance')
                        if feature_importance_df is not None:
                            feature_importance = []
                            for idx, row in feature_importance_df.iterrows():
                                if idx < top_n:
                                    feature_importance.append({
                                        "feature": row.get('Feature', str(idx)),
                                        "feature_korean": self.feature_display_names.get(row.get('Feature', str(idx)), row.get('Feature', str(idx))),
                                        "importance": float(row.get('Importance', 0)),
                                        "std": 0.0
                                    })
                            
                            result = {
                                "method": "pycaret",
                                "feature_importance": feature_importance
                            }
                            self._importance_cache[cache_key] = result
                            return result
                    except:
                        pass
                    
                    # Fallback to model's built-in importance
                    if hasattr(model, '_final_estimator'):
                        final_model = model._final_estimator
                    elif hasattr(model, 'steps'):
                        final_model = model.steps[-1][1]
                    else:
                        final_model = model
                    
                    if hasattr(final_model, 'feature_importances_'):
                        importances = final_model.feature_importances_
                    elif hasattr(final_model, 'coef_'):
                        importances = np.abs(final_model.coef_)
                    else:
                        raise ValueError("Model has no feature importance attribute")
                    
                    feature_importance = []
                    for i, importance in enumerate(importances):
                        if i < len(self.feature_names):
                            english_name = self.feature_names[i]
                            korean_name = self.feature_display_names.get(english_name, english_name)
                            feature_importance.append({
                                "feature": english_name,
                                "feature_korean": korean_name,
                                "importance": float(importance),
                                "std": 0.0
                            })
                    
                    feature_importance.sort(key=lambda x: x["importance"], reverse=True)
                    feature_importance = feature_importance[:top_n]
                    
                    result = {
                        "method": "pycaret",
                        "feature_importance": feature_importance
                    }
                    self._importance_cache[cache_key] = result
                    return result
                    
                except Exception as e:
                    print(f"[WARNING] PyCaret method failed: {str(e)}")
                    
            elif method == "permutation" and SKLEARN_AVAILABLE:
                # Permutation importance
                test_X = X_test if X_test is not None else X_train
                test_y = y_test if y_test is not None else y_train
                
                try:
                    # PyCaret Pipeline 모델 처리
                    if hasattr(model, 'steps'):
                        # Pipeline의 마지막 단계(실제 모델) 추출
                        actual_model = model.steps[-1][1]
                        print(f"[DATA] Using actual model from Pipeline: {type(actual_model).__name__}")
                        
                        # Pipeline 전체로 예측하되, feature importance는 실제 모델에서 추출
                        if hasattr(actual_model, 'coef_'):
                            # Linear 모델인 경우 계수 사용
                            importances = np.abs(actual_model.coef_)
                            feature_importance = []
                            for i, importance in enumerate(importances):
                                if i < len(self.feature_names):
                                    english_name = self.feature_names[i]
                                    korean_name = self.feature_display_names.get(english_name, english_name)
                                    feature_importance.append({
                                        "feature": english_name,
                                        "feature_korean": korean_name,
                                        "importance": float(importance),
                                        "std": 0.0
                                    })
                            feature_importance.sort(key=lambda x: x["importance"], reverse=True)
                            feature_importance = feature_importance[:top_n]
                            
                            result = {
                                "method": "coefficients",
                                "feature_importance": feature_importance
                            }
                            self._importance_cache[cache_key] = result
                            return result
                        elif hasattr(actual_model, 'feature_importances_'):
                            # Tree 기반 모델인 경우
                            importances = actual_model.feature_importances_
                            feature_importance = []
                            for i, importance in enumerate(importances):
                                if i < len(self.feature_names):
                                    english_name = self.feature_names[i]
                                    korean_name = self.feature_display_names.get(english_name, english_name)
                                    feature_importance.append({
                                        "feature": english_name,
                                        "feature_korean": korean_name,
                                        "importance": float(importance),
                                        "std": 0.0
                                    })
                            feature_importance.sort(key=lambda x: x["importance"], reverse=True)
                            feature_importance = feature_importance[:top_n]
                            
                            result = {
                                "method": "built_in",
                                "feature_importance": feature_importance
                            }
                            self._importance_cache[cache_key] = result
                            return result
                    
                    # Pipeline이 아닌 경우 일반적인 permutation importance 계산
                    perm_importance = permutation_importance(model, test_X, test_y, n_repeats=10, random_state=42)
                    
                except Exception as e:
                    print(f"[WARNING] Feature importance calculation failed: {str(e)}")
                    # Fallback: 기본값 반환
                    return {
                        "method": method,
                        "feature_importance": [],
                        "error": str(e)
                    }
                
                for i, importance in enumerate(perm_importance.importances_mean):
                    if i < len(self.feature_names):
                        english_name = self.feature_names[i]
                        korean_name = self.feature_display_names.get(english_name, english_name)
                        feature_importance.append({
                            "feature": english_name,
                            "feature_korean": korean_name,
                            "importance": float(importance),
                            "std": float(perm_importance.importances_std[i])
                        })
                
                feature_importance.sort(key=lambda x: x["importance"], reverse=True)
                feature_importance = feature_importance[:top_n]
            
            elif method == "built_in":
                # 모델의 built-in feature importance
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    for i, importance in enumerate(importances):
                        if i < len(self.feature_names):
                            english_name = self.feature_names[i]
                            korean_name = self.feature_display_names.get(english_name, english_name)
                            feature_importance.append({
                                "feature": english_name,
                                "feature_korean": korean_name,
                                "importance": float(importance)
                            })
                    
                    feature_importance.sort(key=lambda x: x["importance"], reverse=True)
                    feature_importance = feature_importance[:top_n]
                else:
                    raise ValueError("Model does not have built-in feature importance")
            
            result = {
                "message": f"Feature importance analysis completed using {method}",
                "method": method,
                "feature_importance": feature_importance,
                "n_features": len(feature_importance)
            }
            
            # 캐시에 저장
            self._importance_cache[cache_key] = result
            print(f"[DATA] Cached {method} importance for model {model_id}")
            
            return result
            
        except Exception as e:
            logging.error(f"Feature importance analysis failed: {str(e)}")
            return {
                "error": f"Feature importance analysis failed: {str(e)}",
                "method": method,
                "feature_importance": []
            }
    
    def get_lime_analysis(self, model, sample_index: int, num_features: int = 10) -> Dict[str, Any]:
        """LIME 분석 수행"""
        
        if not LIME_AVAILABLE:
            return {
                "error": "LIME not available. Please install with: pip install lime",
                "available": False
            }
        
        try:
            X_train, y_train, X_test, y_test = self._get_training_data()
            
            if X_train is None:
                raise ValueError("No training data available")
            
            print(f"[DATA] LIME Analysis Debug:")
            print(f"   - X_train type: {type(X_train)}")
            print(f"   - X_train shape: {X_train.shape}")
            if hasattr(X_train, 'columns'):
                print(f"   - X_train columns: {list(X_train.columns)}")
            if X_test is not None:
                print(f"   - X_test shape: {X_test.shape}")
                if hasattr(X_test, 'columns'):
                    print(f"   - X_test columns: {list(X_test.columns)}")
            
            # 데이터 준비 (LIME용) - PyCaret 처리 후 실제 컬럼 사용
            if hasattr(X_train, 'values'):
                train_data = X_train.values
                feature_names = X_train.columns.tolist()
                # 한글 컬럼명 매핑 가져오기
                self.feature_display_names = data_service.get_display_names(feature_names)
                print(f"[DATA] LIME using features: {feature_names[:5]}... (총 {len(feature_names)}개)")
            else:
                train_data = X_train
                feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
            
            # 데이터 정규화 및 이상값 처리 (LIME 분포 오류 방지)
            train_data_clean = np.nan_to_num(train_data, nan=0.0, posinf=1e6, neginf=-1e6)
            
            # 각 피처의 분산이 0인 경우 작은 값 추가
            np.random.seed(42)
            for i in range(train_data_clean.shape[1]):
                if np.var(train_data_clean[:, i]) == 0:
                    train_data_clean[:, i] += np.random.normal(0, 1e-6, len(train_data_clean[:, i]))
            
            # 모델을 완전히 래핑하는 클래스 생성
            class WrappedModel:
                def __init__(self, model, feature_names):
                    self.model = model
                    self.feature_names = feature_names
                
                def predict(self, X):
                    try:
                        # numpy 배열을 항상 DataFrame으로 변환
                        if not isinstance(X, pd.DataFrame):
                            if len(X.shape) == 1:
                                X = X.reshape(1, -1)
                            X = pd.DataFrame(X, columns=self.feature_names)
                        return self.model.predict(X)
                    except Exception as e:
                        print(f"[WARNING] WrappedModel prediction error: {e}")
                        # Can't predict for LIME analysis
                        raise ValueError(f"WrappedModel prediction failed: {e}")
            
            wrapped_model = WrappedModel(model, feature_names)
            
            # LIME explainer 생성 (래핑된 모델 사용)
            explainer = lime.lime_tabular.LimeTabularExplainer(
                train_data_clean,
                feature_names=feature_names,
                mode='regression',
                discretize_continuous=False,  # 연속형 변수를 이산화하지 않음
                sample_around_instance=True,  # 인스턴스 주변 샘플링
                random_state=42
            )
            
            # 설명할 인스턴스 선택 (LIME 호환성을 위해 numpy 배열로 변환)
            test_X = X_test if X_test is not None else X_train
            if sample_index >= len(test_X):
                raise ValueError(f"Sample index {sample_index} out of range. Max index: {len(test_X)-1}")
            
            # 인스턴스를 numpy 배열로 변환
            if hasattr(test_X, 'values'):
                test_data = test_X.values
            else:
                test_data = test_X
            
            # 인스턴스 선택 및 정리
            instance = test_data[sample_index]
            instance = np.nan_to_num(instance, nan=0.0, posinf=1e6, neginf=-1e6)
            
            print(f"[DATA] LIME instance debug:")
            print(f"   - Instance shape: {instance.shape}")
            print(f"   - Instance type: {type(instance)}")
            print(f"   - Feature names length: {len(feature_names)}")
            print(f"   - Instance values sample: {instance[:3]}")
            
            # LIME 설명 생성을 위한 완전히 독립적인 예측 함수
            print(f"[DATA] Creating LIME explainer with:")
            print(f"   - Training data shape: {train_data_clean.shape}")
            print(f"   - Feature names: {feature_names}")
            print(f"   - Instance to explain shape: {instance.shape}")
            
            # 래핑된 모델 예측 함수 (LIME 내부 호환성 강화)
            def lime_compatible_predict(X):
                """LIME 내부 호환성을 위한 예측 함수"""
                try:
                    # 입력 데이터 형태 확인 및 정규화
                    if hasattr(X, 'shape'):
                        if len(X.shape) == 1:
                            X = X.reshape(1, -1)
                        print(f"[DATA] LIME internal predict - X shape: {X.shape}")
                    else:
                        X = np.array(X).reshape(1, -1)
                        print(f"[DATA] LIME internal predict - X converted to shape: {X.shape}")
                    
                    # 컬럼 수 검증
                    if X.shape[1] != len(feature_names):
                        print(f"[WARNING] Column mismatch: X has {X.shape[1]} columns, expected {len(feature_names)}")
                        # 컬럼 수가 맞지 않으면 기본값 반환
                        return np.full(X.shape[0], 0.042)
                    
                    # DataFrame 변환 (PyCaret 호환성)
                    X_df = pd.DataFrame(X, columns=feature_names)
                    
                    # PyCaret 모델 예측
                    predictions = wrapped_model.predict(X_df)
                    
                    # 예측 결과 형태 정규화
                    if hasattr(predictions, 'values'):
                        predictions = predictions.values
                    if not isinstance(predictions, np.ndarray):
                        predictions = np.array(predictions)
                    if len(predictions.shape) > 1:
                        predictions = predictions.flatten()
                    
                    print(f"[DATA] LIME prediction successful: {predictions[:3] if len(predictions) > 3 else predictions}")
                    return predictions
                    
                except Exception as e:
                    print(f"[WARNING] LIME prediction error: {e}")
                    # LIME prediction failed
                    raise ValueError(f"LIME prediction function failed: {e}")
            
            # LIME explainer의 설명 생성 시도
            try:
                print(f"[DATA] Starting LIME explain_instance...")
                explanation = explainer.explain_instance(
                    instance, 
                    lime_compatible_predict, 
                    num_features=num_features
                )
                print(f"[DATA] LIME explain_instance completed successfully")
                
            except Exception as lime_error:
                print(f"[WARNING] LIME explain_instance failed: {lime_error}")
                
                # 대체 방법: 더 간단한 LIME 설정으로 재시도
                try:
                    print(f"[DATA] Retrying LIME with simplified settings...")
                    
                    # 더 작은 데이터셋으로 explainer 재생성
                    simple_data = train_data_clean[:100] if len(train_data_clean) > 100 else train_data_clean
                    
                    simple_explainer = lime.lime_tabular.LimeTabularExplainer(
                        simple_data,
                        feature_names=feature_names,
                        mode='regression',
                        discretize_continuous=True,  # 이산화 활성화
                        sample_around_instance=False,  # 단순 샘플링
                        random_state=42
                    )
                    
                    explanation = simple_explainer.explain_instance(
                        instance, 
                        lime_compatible_predict, 
                        num_features=min(num_features, len(feature_names))
                    )
                    print(f"[DATA] LIME retry successful")
                    
                except Exception as retry_error:
                    print(f"[WARNING] LIME retry also failed: {retry_error}")
                    raise ValueError(f"All LIME methods failed. Cannot generate explanation: {str(retry_error)}")
            
            # 설명 결과 파싱 (한글명 포함)
            lime_values = []
            for feature, value in explanation.as_list():
                korean_name = self.feature_display_names.get(feature, feature)
                lime_values.append({
                    "feature": feature,
                    "feature_korean": korean_name,
                    "value": float(value)
                })
            
            # 예측값 (일관성을 위해 wrapped model 사용)
            try:
                instance_df = pd.DataFrame([instance], columns=feature_names)
                prediction = float(wrapped_model.predict(instance_df)[0])
            except Exception as e:
                print(f"[WARNING] Final prediction failed: {e}")
                raise ValueError(f"Model prediction failed for LIME analysis: {str(e)}")
            
            return {
                "message": "LIME analysis completed successfully",
                "available": True,
                "sample_index": sample_index,
                "prediction": prediction,
                "explanation": lime_values,
                "num_features": len(lime_values),
                "intercept": float(explanation.intercept[1]) if hasattr(explanation, 'intercept') else 0
            }
            
        except Exception as e:
            logging.error(f"LIME analysis failed: {str(e)}")
            return {
                "error": f"LIME analysis failed: {str(e)}",
                "available": False
            }
    
    def get_model_performance_analysis(self) -> Dict[str, Any]:
        """모델 성능 분석"""
        
        try:
            X_train, y_train, X_test, y_test = self._get_training_data()
            
            if X_train is None or y_train is None:
                raise ValueError("No training data available for performance analysis")
            
            from app.services.modeling_service import modeling_service
            model = modeling_service.current_model
            
            if model is None:
                raise ValueError("No model available")
            
            # 예측 수행
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test) if X_test is not None else None
            
            # 성능 메트릭 계산
            performance = {
                "train_metrics": {
                    "mse": float(mean_squared_error(y_train, train_pred)),
                    "mae": float(mean_absolute_error(y_train, train_pred)),
                    "r2": float(r2_score(y_train, train_pred))
                }
            }
            
            if test_pred is not None and y_test is not None:
                performance["test_metrics"] = {
                    "mse": float(mean_squared_error(y_test, test_pred)),
                    "mae": float(mean_absolute_error(y_test, test_pred)),
                    "r2": float(r2_score(y_test, test_pred))
                }
            
            # 잔차 분석
            train_residuals = y_train - train_pred
            performance["residual_analysis"] = {
                "mean_residual": float(np.mean(train_residuals)),
                "std_residual": float(np.std(train_residuals)),
                "residual_range": [float(np.min(train_residuals)), float(np.max(train_residuals))]
            }
            
            return {
                "message": "Model performance analysis completed",
                "performance": performance,
                "model_type": type(model).__name__
            }
            
        except Exception as e:
            logging.error(f"Performance analysis failed: {str(e)}")
            return {
                "error": f"Performance analysis failed: {str(e)}",
                "performance": {}
            }
    
    def get_partial_dependence(self, model, feature_name: str, num_grid_points: int = 50) -> Dict[str, Any]:
        """부분 의존성 플롯 데이터 생성"""
        
        if not SKLEARN_AVAILABLE:
            return {
                "error": "scikit-learn not available for partial dependence analysis",
                "available": False
            }
        
        try:
            X_train, y_train, X_test, y_test = self._get_training_data()
            
            if X_train is None:
                raise ValueError("No training data available")
            
            if feature_name not in X_train.columns:
                raise ValueError(f"Feature '{feature_name}' not found in training data")
            
            feature_idx = list(X_train.columns).index(feature_name)
            
            # Partial dependence 계산
            pd_results = partial_dependence(
                model, X_train, [feature_idx], 
                grid_resolution=num_grid_points
            )
            
            grid_values = pd_results[1][0]
            pd_values = pd_results[0][0]
            
            return {
                "message": "Partial dependence analysis completed",
                "feature_name": feature_name,
                "grid_values": grid_values.tolist(),
                "partial_dependence": pd_values.tolist(),
                "num_points": len(grid_values)
            }
            
        except Exception as e:
            logging.error(f"Partial dependence analysis failed: {str(e)}")
            return {
                "error": f"Partial dependence analysis failed: {str(e)}",
                "available": False
            }
    
    def get_residual_analysis(self, model) -> Dict[str, Any]:
        """잔차 분석"""
        
        try:
            X_train, y_train, X_test, y_test = self._get_training_data()
            
            if X_train is None or y_train is None:
                raise ValueError("No training data available")
            
            # 예측 및 잔차 계산
            train_pred = model.predict(X_train)
            residuals = y_train - train_pred
            
            # 잔차 통계
            residual_stats = {
                "mean": float(np.mean(residuals)),
                "std": float(np.std(residuals)),
                "min": float(np.min(residuals)),
                "max": float(np.max(residuals)),
                "q25": float(np.percentile(residuals, 25)),
                "q50": float(np.percentile(residuals, 50)),
                "q75": float(np.percentile(residuals, 75))
            }
            
            # 정규성 검정 (간단한 버전)
            normalized_residuals = (residuals - np.mean(residuals)) / np.std(residuals)
            
            return {
                "message": "Residual analysis completed",
                "residual_statistics": residual_stats,
                "residuals": residuals.tolist()[:100],  # 처음 100개만
                "predictions": train_pred.tolist()[:100],
                "actuals": y_train.tolist()[:100] if hasattr(y_train, 'tolist') else list(y_train)[:100]
            }
            
        except Exception as e:
            logging.error(f"Residual analysis failed: {str(e)}")
            return {
                "error": f"Residual analysis failed: {str(e)}"
            }
    
    def get_prediction_intervals(self, model, confidence_level: float = 0.95) -> Dict[str, Any]:
        """예측 구간 계산"""
        
        try:
            X_train, y_train, X_test, y_test = self._get_training_data()
            
            if X_train is None or y_train is None:
                raise ValueError("No training data available")
            
            # 예측 수행
            predictions = model.predict(X_test if X_test is not None else X_train)
            
            # 잔차 기반 예측 구간 (간단한 방법)
            train_pred = model.predict(X_train)
            residuals = y_train - train_pred
            residual_std = np.std(residuals)
            
            # 신뢰구간 계산
            from scipy import stats
            alpha = 1 - confidence_level
            z_score = stats.norm.ppf(1 - alpha/2)
            
            margin_of_error = z_score * residual_std
            
            lower_bound = predictions - margin_of_error
            upper_bound = predictions + margin_of_error
            
            return {
                "message": "Prediction intervals calculated",
                "confidence_level": confidence_level,
                "predictions": predictions.tolist()[:100],
                "lower_bound": lower_bound.tolist()[:100],
                "upper_bound": upper_bound.tolist()[:100],
                "margin_of_error": float(margin_of_error)
            }
            
        except Exception as e:
            logging.error(f"Prediction intervals calculation failed: {str(e)}")
            return {
                "error": f"Prediction intervals calculation failed: {str(e)}"
            }
    
    def get_contribution_plot(self, model, sample_index: Optional[int] = None, top_n: int = 10) -> Dict[str, Any]:
        """개별 예측에 대한 Feature Contribution 플롯 데이터"""
        
        try:
            # PyCaret 환경이 필요한지 확인
            try:
                from pycaret.regression import get_config
                has_pycaret_env = True
                print("[DATA] PyCaret environment available")
            except:
                has_pycaret_env = False
                print("[WARNING] PyCaret environment not available, using fallback method")
            
            # 모델에서 실제 feature 이름 가져오기
            feature_names = None
            try:
                # PyCaret 모델에서 feature 이름 가져오기
                if hasattr(model, 'feature_names_in_'):
                    feature_names = list(model.feature_names_in_)
                    print(f"[OK] Got feature names from model.feature_names_in_: {len(feature_names)} features")
                elif hasattr(model, 'steps') and len(model.steps) > 0:
                    # Pipeline인 경우
                    final_model = model.steps[-1][1]
                    if hasattr(final_model, 'feature_names_in_'):
                        feature_names = list(final_model.feature_names_in_)
                        print(f"[OK] Got feature names from pipeline final model: {len(feature_names)} features")
                
                if feature_names:
                    print(f"[SEARCH] Model features: {feature_names[:10]}..." if len(feature_names) > 10 else f"[SEARCH] Model features: {feature_names}")
                else:
                    print("[WARNING] Could not get feature names from model")
            except Exception as e:
                print(f"[WARNING] Error getting feature names: {str(e)}")
                feature_names = None
            
            # 2026년 예측을 위한 데이터 가져오기 - data_service의 마스터 데이터 사용
            try:
                from app.services.data_service import data_service
                
                # data_service에서 2025년 데이터 가져오기 (2026년 예측용)
                master_data = data_service.master_data
                if master_data is not None and len(master_data) > 0:
                    print("[DATA] Using 2025 data from data_service for 2026 contribution analysis")
                    
                    # 2025년 데이터 (마지막 행) 사용 - 2026년 예측을 위함
                    prediction_data = master_data.iloc[[-1]].copy()  # 마지막 행 사용
                    
                    # 데이터 누수 방지: 임금 관련 컬럼 모두 제거
                    wage_columns_to_remove = [
                        'wage_increase_total_sbl', 'wage_increase_mi_sbl', 'wage_increase_bu_sbl',
                        'wage_increase_baseup_sbl', 'Base-up 인상률', '성과인상률', '임금인상률',
                        'wage_increase_total_group', 'wage_increase_mi_group', 'wage_increase_bu_group'
                    ]
                    prediction_data = prediction_data.drop(columns=wage_columns_to_remove, errors='ignore')
                    
                    # year 컬럼도 제거 (모델링에서 사용하지 않음)
                    if 'eng' in prediction_data.columns:
                        prediction_data = prediction_data.drop(columns=['eng'])
                    
                    # 모델이 훈련된 정확한 feature names 사용
                    if feature_names and len(feature_names) > 0:
                        # 모델이 기대하는 feature만 사용 (순서도 정확히)
                        available_features = [f for f in feature_names if f in prediction_data.columns]
                        if len(available_features) == len(feature_names):
                            prediction_data = prediction_data[feature_names]
                            print(f"[DATA] Using model's exact features for 2026 prediction: {len(feature_names)} features")
                        else:
                            print(f"[WARNING] Missing some model features, available: {len(available_features)}/{len(feature_names)}")
                            print(f"[WARNING] Missing features: {set(feature_names) - set(available_features)}")
                    
                    print(f"[DATA] 2026 Prediction data shape: {prediction_data.shape}")
                    print(f"[SEARCH] 2026 Prediction data columns: {list(prediction_data.columns)}")
                else:
                    # Fallback: PyCaret 훈련 데이터 사용
                    print("[WARNING] No master data from data_service, trying PyCaret training data")
                    X_train, _, _, _ = self._get_training_data()
                    if X_train is not None and len(X_train) > 0:
                        if feature_names and len(feature_names) > 0:
                            prediction_data = X_train[feature_names].iloc[-1:].copy()
                        else:
                            prediction_data = X_train.iloc[-1:].copy()
                        print(f"[DATA] Using training data shape: {prediction_data.shape}")
                    else:
                        raise ValueError("No training data available")
                        
            except Exception as e:
                print(f"[WARNING] Could not get prediction data: {str(e)}")
                raise ValueError(f"Cannot get prediction data for contribution analysis: {str(e)}")
            
            print(f"[DATA] Using prediction data with shape: {prediction_data.shape}")
            print(f"[SEARCH] Prediction data columns: {list(prediction_data.columns)}")
            
            # 모델 예측
            prediction = model.predict(prediction_data)[0]
            print(f"[SEARCH] Model prediction: {prediction}")
            
            # 기준선 계산 - 훈련 데이터의 실제 평균 사용
            try:
                _, y_train, _, _ = self._get_training_data()
                if y_train is not None:
                    baseline_prediction = float(y_train.mean())
                    print(f"[DATA] Using actual training data mean as baseline: {baseline_prediction}")
                else:
                    raise ValueError("No training target data available for baseline calculation")
            except Exception as e:
                raise ValueError(f"Cannot calculate baseline prediction from training data: {str(e)}")
            
            # Linear 모델인 경우 계수 기반 기여도 계산
            if hasattr(model, 'steps') and len(model.steps) > 0:
                # Pipeline의 마지막 단계가 실제 모델
                actual_model = model.steps[-1][1]
                print(f"[DATA] Using model from pipeline: {type(actual_model).__name__}")
            else:
                actual_model = model
                print(f"[DATA] Using direct model: {type(actual_model).__name__}")
            
            contributions = []
            
            if hasattr(actual_model, 'coef_'):
                # Linear 모델의 계수 기반 기여도
                coefficients = actual_model.coef_
                sample_values = prediction_data.iloc[0].values
                
                print(f"[DATA] Model coefficients shape: {coefficients.shape}")
                print(f"[DATA] Sample values shape: {sample_values.shape}")
                
                for i, (coef, value) in enumerate(zip(coefficients, sample_values)):
                    if i < len(prediction_data.columns):
                        feature_name = prediction_data.columns[i]
                        contribution = coef * (value - 0)  # contribution = coefficient * (value - baseline)
                        
                        # 한국어 이름 가져오기
                        korean_name = self.feature_display_names.get(feature_name, feature_name)
                        
                        contributions.append({
                            "feature": feature_name,
                            "feature_korean": korean_name,
                            "feature_value": float(value),
                            "contribution": float(contribution),
                            "contribution_abs": float(abs(contribution))
                        })
                
                # 기여도 크기순으로 정렬하고 상위 N개 선택
                contributions.sort(key=lambda x: x["contribution_abs"], reverse=True)
                contributions = contributions[:top_n]
                
                # 총 기여도 계산
                total_contribution = sum(c["contribution"] for c in contributions)
                
                return {
                    "message": "Feature contribution analysis completed successfully",
                    "sample_index": 0,  # 2026년 예측 데이터
                    "prediction": float(prediction),
                    "actual_value": float(baseline_prediction),  # 기준값 사용
                    "baseline_prediction": float(baseline_prediction),
                    "total_contribution": float(total_contribution),
                    "residual": float(prediction - baseline_prediction),
                    "contributions": contributions,
                    "n_features": len(contributions)
                }
            
            else:
                # Tree 기반 모델이나 다른 모델의 경우 permutation 기반 방법 사용
                print("[DATA] Using permutation-based contribution analysis")
                
                # 각 feature를 평균값으로 대체해가며 예측 변화 측정
                baseline_values = prediction_data.mean()
                
                for i, feature_name in enumerate(prediction_data.columns):
                    if i >= top_n:
                        break
                        
                    # feature를 평균값으로 대체한 데이터
                    modified_data = prediction_data.copy()
                    modified_data.iloc[0, i] = baseline_values.iloc[i]
                    
                    # 예측값 변화 계산
                    modified_prediction = model.predict(modified_data)[0]
                    contribution = prediction - modified_prediction
                    
                    # 한국어 이름 가져오기
                    korean_name = self.feature_display_names.get(feature_name, feature_name)
                    
                    contributions.append({
                        "feature": feature_name,
                        "feature_korean": korean_name,
                        "feature_value": float(prediction_data.iloc[0, i]),
                        "contribution": float(contribution),
                        "contribution_abs": float(abs(contribution))
                    })
                
                # 기여도 크기순으로 정렬
                contributions.sort(key=lambda x: x["contribution_abs"], reverse=True)
                
                # 총 기여도 계산
                total_contribution = sum(c["contribution"] for c in contributions)
                
                return {
                    "message": "Feature contribution analysis completed successfully (permutation method)",
                    "sample_index": 0,  # 2026년 예측 데이터
                    "prediction": float(prediction),
                    "actual_value": float(baseline_prediction),  # 기준값 사용
                    "baseline_prediction": float(baseline_prediction),
                    "total_contribution": float(total_contribution),
                    "residual": float(prediction - baseline_prediction),
                    "contributions": contributions,
                    "n_features": len(contributions)
                }
        
        except Exception as e:
            print(f"[ERROR] Contribution plot analysis failed: {str(e)}")
            return {
                "error": f"Contribution analysis failed: {str(e)}",
                "available": False
            }

# 싱글톤 인스턴스
analysis_service = AnalysisService()