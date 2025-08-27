import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import warnings
import io
import sys
import os
import pickle
from contextlib import redirect_stdout, redirect_stderr
import logging

# PyCaret 라이브러리 import with error handling
try:
    from pycaret.regression import (
        setup, compare_models, create_model, tune_model, 
        finalize_model, predict_model, evaluate_model, 
        pull, get_config
    )
    PYCARET_AVAILABLE = True
except ImportError:
    PYCARET_AVAILABLE = False
    logging.warning("PyCaret not available. Modeling functionality will be limited.")

from app.services.data_service import data_service

class ModelingService:
    def __init__(self):
        self.current_experiment = None
        self.current_model = None
        self.baseup_model = None  # Base-up 전용 모델
        self.performance_model = None  # 성과급 전용 모델
        self.model_results = None
        self.is_setup_complete = False
        self.is_model_trained_individually = False  # 개별 모델 학습 여부
        self.current_target = None  # 현재 타겟 컬럼
        
        # Feature importance 저장
        self.baseup_feature_importance = None
        self.performance_feature_importance = None
        self.current_feature_importance = None
        
        # 데이터 크기에 따른 모델 선택
        # PDF 분석 결과를 반영하여 Random Forest를 소규모 데이터에도 포함
        # Lasso 제외 (적은 데이터에서 모든 계수를 0으로 만드는 문제 방지)
        self.small_data_models = ['lr', 'ridge', 'en', 'dt', 'rf']  # lasso 제외
        self.medium_data_models = ['lr', 'ridge', 'en', 'dt', 'rf', 'gbr']  # lasso 제외
        self.large_data_models = ['lr', 'ridge', 'en', 'dt', 'rf', 'gbr', 'xgboost', 'lightgbm']  # lasso 제외
        
        # 모델 저장 경로
        self.model_dir = "saved_models"
        self.baseup_model_path = os.path.join(self.model_dir, "baseup_model.pkl")
        self.performance_model_path = os.path.join(self.model_dir, "performance_model.pkl")
        
        # 서버 시작 시 저장된 모델 자동 로드
        print("\n" + "=" * 80)
        print("🚀 INITIALIZING MODEL SERVICE")
        print("=" * 80)
        self.load_saved_models()
        print("=" * 80 + "\n")
    
    def check_pycaret_availability(self) -> bool:
        """PyCaret 사용 가능 여부 확인"""
        return PYCARET_AVAILABLE
    
    def get_optimal_settings(self, data_size: int) -> Dict[str, Any]:
        """데이터 크기에 따른 최적 설정 반환"""
        if data_size < 30:
            return {
                'train_size': 0.9,
                'cv_folds': 3,
                'models': self.small_data_models,
                'normalize': True,
                'transformation': False,
                'remove_outliers': False,
                'feature_selection': False,
                'n_features_to_select': 0.8
            }
        elif data_size < 100:
            return {
                'train_size': 0.8,
                'cv_folds': 5,
                'models': self.medium_data_models,
                'normalize': True,
                'transformation': True,
                'remove_outliers': True,
                'feature_selection': True,
                'n_features_to_select': 0.7
            }
        else:
            return {
                'train_size': 0.7,
                'cv_folds': 10,
                'models': self.large_data_models,
                'normalize': True,
                'transformation': True,
                'remove_outliers': True,
                'feature_selection': True,
                'n_features_to_select': 0.6
            }
    
    def prepare_data_for_modeling(self, target_column: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """모델링을 위한 데이터 준비"""
        if data_service.current_data is None:
            raise ValueError("No data loaded for modeling")
        
        df = data_service.current_data.copy()
        
        # 기본 데이터 정리
        info = {
            'original_shape': df.shape,
            'target_column': target_column,
            'numeric_columns': [],
            'categorical_columns': [],
            'dropped_columns': []
        }
        
        # 타겟 컬럼 존재 확인
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")
        
        # 타겟 컬럼에 결측값이 있는 행 제거 (2025년 예측 대상 데이터 제외)
        initial_rows = len(df)
        df = df.dropna(subset=[target_column])
        removed_rows = initial_rows - len(df)
        
        if removed_rows > 0:
            info['removed_target_missing'] = removed_rows
            print(f"📊 Removed {removed_rows} rows with missing target values (likely future prediction data)")
        
        # 타겟 컬럼이 충분한 데이터가 있는지 확인
        if len(df) < 5:
            raise ValueError(f"Insufficient training data: only {len(df)} rows with valid target values")
        
        # 최소한의 전처리만 수행 (PyCaret이 나머지를 처리)
        # '-' 값을 NaN으로 변환 (PyCaret이 인식할 수 있도록)
        df = df.replace(['-', ''], np.nan)
        
        # 범주형으로 보이는 숫자 컬럼을 실제 숫자로 변환
        for col in df.columns:
            if col != target_column:
                try:
                    # 숫자로 변환 가능한 컬럼은 변환
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except:
                    pass
        
        # 년도 컬럼 제거 (시계열 인덱스이므로 피처에서 제외)
        year_columns = ['year', 'Year', 'YEAR', '년도', '연도']
        for year_col in year_columns:
            if year_col in df.columns and year_col != target_column:
                df = df.drop(columns=[year_col])
                info['dropped_columns'].append(year_col)
                print(f"📊 Removed year column: {year_col}")
        
        # 타겟 컬럼이 숫자형인지 확인
        try:
            df[target_column] = pd.to_numeric(df[target_column], errors='coerce')
            df = df.dropna(subset=[target_column])  # 변환 실패한 행 제거
        except:
            raise ValueError(f"Target column '{target_column}' must contain numeric values")
        
        # PyCaret이 모든 컬럼을 처리하도록 함
        # 기본 정보만 수집
        for col in df.columns:
            if col != target_column:
                if pd.api.types.is_numeric_dtype(df[col]):
                    info['numeric_columns'].append(col)
                else:
                    info['categorical_columns'].append(col)
        
        # 최종 정리
        info['final_shape'] = df.shape
        info['feature_count'] = len(df.columns) - 1
        
        return df, info
    
    def setup_pycaret_environment(
        self, 
        target_column: str, 
        train_size: Optional[float] = None,
        session_id: int = 42
    ) -> Dict[str, Any]:
        """PyCaret 환경 설정"""
        
        # session_id로 충분함 - 추가 seed 설정 제거
        
        if not self.check_pycaret_availability():
            raise RuntimeError("PyCaret is not available. Please install it first.")
        
        # 데이터 준비
        ml_data, data_info = self.prepare_data_for_modeling(target_column)
        
        # 최적 설정 가져오기
        optimal_settings = self.get_optimal_settings(len(ml_data))
        actual_train_size = train_size or optimal_settings['train_size']
        
        # 출력 억제를 위한 설정
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        
        try:
            # 모든 출력 억제
            sys.stdout = io.StringIO()
            sys.stderr = io.StringIO()
            
            # PyCaret setup 실행 (자동 전처리 강화)
            exp = setup(
                data=ml_data,
                target=target_column,
                session_id=session_id,
                train_size=actual_train_size,
                html=False,
                verbose=False,
                
                # 자동 데이터 타입 추론 및 전처리
                numeric_features=None,  # PyCaret이 자동 감지
                categorical_features=None,  # PyCaret이 자동 감지
                ignore_features=None,
                
                # 결측값 처리
                imputation_type='simple',  # 단순 대체
                numeric_imputation='mean',  # 숫자형: 평균값
                categorical_imputation='mode',  # 범주형: 최빈값
                
                # 적응적 전처리 옵션
                normalize=optimal_settings['normalize'],
                transformation=optimal_settings['transformation'],
                remove_outliers=optimal_settings['remove_outliers'],
                remove_multicollinearity=True,
                multicollinearity_threshold=0.9,
                feature_selection=optimal_settings['feature_selection'],
                n_features_to_select=optimal_settings['n_features_to_select'],
                
                # Feature 생성 설정
                polynomial_features=False,  # 다항식 feature 생성 비활성화 (feature 이름 충돌 방지)
                polynomial_degree=2,  # 다항식 차수 (사용 안 함)
                
                # CV 전략
                fold_strategy='kfold',
                fold=optimal_settings['cv_folds']
            )
            
            self.current_experiment = exp
            self.is_setup_complete = True
            self.current_target = target_column  # 현재 타겟 저장
            
        except Exception as e:
            raise RuntimeError(f"PyCaret setup failed: {str(e)}")
        finally:
            # 출력 복원
            sys.stdout = old_stdout
            sys.stderr = old_stderr
        
        # 설정 정보 반환
        return {
            'message': 'PyCaret environment setup completed successfully',
            'data_info': data_info,
            'optimal_settings': optimal_settings,
            'train_size': actual_train_size,
            'available_models': optimal_settings['models']
        }
    
    def compare_models_adaptive(self, n_select: int = 3) -> Dict[str, Any]:
        """데이터 크기에 적응적인 모델 비교"""
        
        if not self.is_setup_complete:
            raise RuntimeError("PyCaret environment not setup. Call setup_pycaret_environment first.")
        
        # PyCaret이 자체적으로 seed를 관리하도록 함
        
        # 현재 데이터 크기 확인
        data_size = len(data_service.current_data)
        optimal_settings = self.get_optimal_settings(data_size)
        models_to_use = optimal_settings['models']
        
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        
        # 디버깅 출력 (stdout 억제 전에)
        print(f"📊 Comparing models: {models_to_use}")
        print(f"📊 Current target: {self.current_target}")
        
        try:
            # 출력 억제
            sys.stdout = io.StringIO()
            sys.stderr = io.StringIO()
            
            # 모델 비교 실행
            # 작은 데이터셋에서는 MAE가 더 신뢰할 수 있는 지표
            best_models = compare_models(
                include=models_to_use,
                sort='MAE',  # MAE가 낮을수록 좋음 (R2는 음수가 나올 수 있음)
                n_select=min(n_select, len(models_to_use)),
                verbose=False,
                fold=3  # 빠른 비교를 위해 fold 수 제한
            )
            
            # 단일 모델이 반환된 경우 리스트로 변환
            if not isinstance(best_models, list):
                best_models = [best_models]
            
            # 결과 정보 추출
            comparison_results = pull()
            
            # stdout 복원 후 디버깅 출력
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            print(f"📊 Pull results shape: {comparison_results.shape if hasattr(comparison_results, 'shape') else 'N/A'}")
            print(f"📊 Pull results columns: {list(comparison_results.columns) if hasattr(comparison_results, 'columns') else 'N/A'}")
            if not comparison_results.empty:
                print(f"📊 Top model from pull: {comparison_results.index[0]}")
            
            # 다시 억제
            sys.stdout = io.StringIO()
            sys.stderr = io.StringIO()
            
            self.model_results = {
                'best_models': best_models,
                'comparison_df': comparison_results,
                'recommended_model': best_models[0] if best_models else None
            }
            
            # current_model 설정
            self.current_model = best_models[0] if best_models else None
            
            # Feature importance 캡처
            if self.current_model:
                feature_importance = self._capture_feature_importance(self.current_model)
                self.current_feature_importance = feature_importance
            else:
                feature_importance = None
            
            # 타겟에 따른 모델 및 feature importance 저장
            if self.current_target == 'wage_increase_bu_sbl':
                self.baseup_model = self.current_model
                self.baseup_feature_importance = feature_importance
                print(f"✅ Base-up model saved: {type(self.current_model).__name__} with {len(feature_importance) if feature_importance else 0} features")
            elif self.current_target == 'wage_increase_mi_sbl':
                self.performance_model = self.current_model
                self.performance_feature_importance = feature_importance
                print(f"✅ Performance model saved: {type(self.current_model).__name__} with {len(feature_importance) if feature_importance else 0} features")
            
        except Exception as e:
            # 실패 시 기본 선형 회귀 사용
            warnings.warn(f"Model comparison failed: {str(e)}. Using default linear regression.")
            
            linear_model = create_model('lr', verbose=False)
            self.model_results = {
                'best_models': [linear_model],
                'comparison_df': None,
                'recommended_model': linear_model,
                'fallback_used': True
            }
            self.current_model = linear_model
            
        finally:
            # 출력 복원
            sys.stdout = old_stdout
            sys.stderr = old_stderr
        
        # comparison_df를 JSON으로 변환
        comparison_results = []
        if self.model_results['comparison_df'] is not None:
            df = self.model_results['comparison_df']
            # Model 열이 있는 경우 사용, 없으면 인덱스 사용
            if 'Model' in df.columns:
                for _, row in df.iterrows():
                    comparison_results.append({
                        'Model': str(row['Model']),
                        'MAE': float(row.get('MAE', 0)) if 'MAE' in row else None,
                        'MSE': float(row.get('MSE', 0)) if 'MSE' in row else None,
                        'RMSE': float(row.get('RMSE', 0)) if 'RMSE' in row else None,
                        'R2': float(row.get('R2', 0)) if 'R2' in row else None,
                        'RMSLE': float(row.get('RMSLE', 0)) if 'RMSLE' in row else None,
                        'MAPE': float(row.get('MAPE', 0)) if 'MAPE' in row else None
                    })
            else:
                # Model 열이 없으면 인덱스를 모델명으로 사용
                for idx, row in df.iterrows():
                    comparison_results.append({
                        'Model': idx if isinstance(idx, str) else str(idx),
                        'MAE': float(row.get('MAE', 0)) if 'MAE' in row else None,
                        'MSE': float(row.get('MSE', 0)) if 'MSE' in row else None,
                        'RMSE': float(row.get('RMSE', 0)) if 'RMSE' in row else None,
                        'R2': float(row.get('R2', 0)) if 'R2' in row else None,
                        'RMSLE': float(row.get('RMSLE', 0)) if 'RMSLE' in row else None,
                        'MAPE': float(row.get('MAPE', 0)) if 'MAPE' in row else None
                    })
        
        return {
            'message': 'Model comparison completed',
            'models_compared': len(models_to_use),
            'best_model_count': len(self.model_results['best_models']),
            'recommended_model_type': type(self.model_results['recommended_model']).__name__,
            'comparison_available': self.model_results['comparison_df'] is not None,
            'comparison_results': comparison_results,
            'data_size_category': 'small' if data_size < 30 else 'medium' if data_size < 100 else 'large'
        }
    
    def train_specific_model(self, model_name: str) -> Dict[str, Any]:
        """특정 모델 학습"""
        
        if not self.is_setup_complete:
            raise RuntimeError("PyCaret environment not setup. Call setup_pycaret_environment first.")
        
        # 모델 이름 매핑 (전체 이름 -> 코드)
        model_name_map = {
            'Linear Regression': 'lr',
            'Ridge Regression': 'ridge',
            'Lasso Regression': 'lasso',
            'Elastic Net': 'en',
            'Decision Tree Regressor': 'dt',
            'Random Forest Regressor': 'rf',
            'Gradient Boosting Regressor': 'gbr',
            'XGBoost Regressor': 'xgboost',
            'Light Gradient Boosting Machine': 'lightgbm',
            # 코드도 그대로 받을 수 있도록
            'lr': 'lr',
            'ridge': 'ridge',
            'lasso': 'lasso',
            'en': 'en',
            'dt': 'dt',
            'rf': 'rf',
            'gbr': 'gbr',
            'xgboost': 'xgboost',
            'lightgbm': 'lightgbm'
        }
        
        # 모델 이름을 코드로 변환
        model_code = model_name_map.get(model_name, model_name.lower())
        print(f"📊 Training model: {model_name} -> {model_code}")
        
        # PyCaret이 자체적으로 seed를 관리하도록 함
        
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        
        try:
            # 출력 억제
            sys.stdout = io.StringIO()
            sys.stderr = io.StringIO()
            
            # 모델 생성
            model = create_model(model_code, verbose=False)
            
            # 모델 튜닝 (선택적)
            try:
                tuned_model = tune_model(model, optimize='R2', verbose=False)
            except:
                tuned_model = model
            
            # 최종 모델
            try:
                final_model = finalize_model(tuned_model)
            except:
                final_model = tuned_model
            
            self.current_model = final_model
            self.is_model_trained_individually = True  # 개별 모델 학습 완료
            
            # Feature importance 캡처
            feature_importance = self._capture_feature_importance(final_model)
            
            # 타겟에 따라 모델 및 feature importance 저장
            if self.current_target == 'wage_increase_bu_sbl':
                self.baseup_model = final_model
                self.baseup_feature_importance = feature_importance
                logging.info(f"Base-up model stored with {len(feature_importance) if feature_importance else 0} features")
            elif self.current_target == 'wage_increase_mi_sbl':
                self.performance_model = final_model
                self.performance_feature_importance = feature_importance
                logging.info(f"Performance model stored with {len(feature_importance) if feature_importance else 0} features")
            
            self.current_feature_importance = feature_importance
            
            # 모델 평가 메트릭 가져오기
            try:
                # 현재 모델의 성능 평가
                from pycaret.regression import predict_model, pull
                predictions = predict_model(final_model, verbose=False)
                metrics = pull()
                
                # 메트릭 추출
                model_metrics = {}
                if metrics is not None and not metrics.empty:
                    if 'MAE' in metrics.columns:
                        model_metrics['MAE'] = float(metrics['MAE'].iloc[-1])
                    if 'MSE' in metrics.columns:
                        model_metrics['MSE'] = float(metrics['MSE'].iloc[-1])
                    if 'RMSE' in metrics.columns:
                        model_metrics['RMSE'] = float(metrics['RMSE'].iloc[-1])
                    if 'R2' in metrics.columns:
                        model_metrics['R2'] = float(metrics['R2'].iloc[-1])
                    if 'RMSLE' in metrics.columns:
                        model_metrics['RMSLE'] = float(metrics['RMSLE'].iloc[-1])
                    if 'MAPE' in metrics.columns:
                        model_metrics['MAPE'] = float(metrics['MAPE'].iloc[-1])
            except:
                model_metrics = {}
            
        except Exception as e:
            raise RuntimeError(f"Model training failed: {str(e)}")
        finally:
            # 출력 복원
            sys.stdout = old_stdout
            sys.stderr = old_stderr
        
        return {
            'message': f'Model {model_name} trained successfully',
            'model_type': type(self.current_model).__name__,
            'model_name': model_name,
            'metrics': model_metrics if model_metrics else None
        }
    
    def get_model_evaluation(self) -> Dict[str, Any]:
        """현재 모델의 평가 결과 반환"""
        
        if self.current_model is None:
            if self.model_results and self.model_results['recommended_model']:
                self.current_model = self.model_results['recommended_model']
            else:
                raise RuntimeError("No trained model available")
        
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        
        try:
            # 출력 억제
            sys.stdout = io.StringIO()
            sys.stderr = io.StringIO()
            
            # 모델 평가
            evaluate_model(self.current_model)
            evaluation_results = pull()
            
        except Exception as e:
            # 평가 실패 시 기본 정보만 반환
            evaluation_results = None
            warnings.warn(f"Model evaluation failed: {str(e)}")
        finally:
            # 출력 복원
            sys.stdout = old_stdout
            sys.stderr = old_stderr
        
        return {
            'message': 'Model evaluation completed',
            'model_type': type(self.current_model).__name__,
            'evaluation_available': evaluation_results is not None,
            'evaluation_data': evaluation_results.to_dict() if evaluation_results is not None else None
        }
    
    def predict_with_model(self, prediction_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """모델을 사용하여 예측 수행"""
        
        if self.current_model is None:
            raise RuntimeError("No trained model available for prediction")
        
        if prediction_data is None:
            # 테스트 데이터로 예측
            try:
                old_stdout = sys.stdout
                sys.stdout = io.StringIO()
                
                predictions = predict_model(self.current_model)
                prediction_results = pull()
                
            except Exception as e:
                raise RuntimeError(f"Prediction failed: {str(e)}")
            finally:
                sys.stdout = old_stdout
        else:
            # 사용자 제공 데이터로 예측
            try:
                old_stdout = sys.stdout
                old_stderr = sys.stderr
                sys.stdout = io.StringIO()
                sys.stderr = io.StringIO()
                
                # PyCaret의 TransformerWrapper를 사용하여 변환 파이프라인 가져오기
                try:
                    # get_config를 사용하여 현재 실험의 변환 파이프라인 가져오기
                    X_train = get_config('X_train')
                    
                    # 학습 시 사용된 feature 이름 가져오기
                    if hasattr(X_train, 'columns'):
                        expected_features = X_train.columns.tolist()
                    else:
                        expected_features = None
                    
                    # prediction_data의 컬럼을 학습 시 사용된 feature에 맞게 조정
                    if expected_features:
                        # 필요한 컬럼만 선택하고 순서 맞추기
                        available_cols = [col for col in expected_features if col in prediction_data.columns]
                        prediction_data_aligned = prediction_data[available_cols].copy()
                        
                        # 누락된 컬럼이 있으면 0으로 채우기 (polynomial features 등)
                        for col in expected_features:
                            if col not in prediction_data_aligned.columns:
                                prediction_data_aligned[col] = 0
                        
                        # 컬럼 순서 맞추기
                        prediction_data_aligned = prediction_data_aligned[expected_features]
                    else:
                        prediction_data_aligned = prediction_data
                    
                    predictions = predict_model(self.current_model, data=prediction_data_aligned)
                    
                except Exception as align_error:
                    # 정렬 실패 시 원본 데이터로 예측 시도
                    warnings.warn(f"Feature alignment failed: {str(align_error)}. Trying with original data.")
                    predictions = predict_model(self.current_model, data=prediction_data)
                
            except Exception as e:
                raise RuntimeError(f"Prediction with custom data failed: {str(e)}")
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr
            
            prediction_results = None
        
        return {
            'message': 'Prediction completed successfully',
            'predictions_available': predictions is not None,
            'prediction_count': len(predictions) if predictions is not None else 0,
            'predictions': predictions.to_dict(orient='records') if predictions is not None else None,
            'evaluation_metrics': prediction_results.to_dict() if prediction_results is not None else None
        }
    
    def get_modeling_status(self) -> Dict[str, Any]:
        """현재 모델링 상태 반환"""
        return {
            'pycaret_available': self.check_pycaret_availability(),
            'environment_setup': self.is_setup_complete,
            'model_trained': self.is_model_trained_individually,  # 개별 학습 여부로 변경
            'models_compared': self.model_results is not None,
            'data_loaded': data_service.current_data is not None,
            'current_model_type': type(self.current_model).__name__ if self.current_model else None,
            'has_model': self.current_model is not None  # 모델 존재 여부
        }
    
    def clear_models(self) -> Dict[str, Any]:
        """모든 모델 및 실험 초기화"""
        self.current_experiment = None
        self.current_model = None
        self.model_results = None
        self.is_setup_complete = False
        self.is_model_trained_individually = False  # 개별 학습 상태도 초기화
        self.baseup_feature_importance = None
        self.performance_feature_importance = None
        self.current_feature_importance = None
        
        return {
            'message': 'All models and experiments cleared successfully'
        }
    
    def _capture_feature_importance(self, model) -> List[Dict[str, Any]]:
        """모델의 feature importance를 캡처하는 내부 메서드"""
        importance_list = []
        print(f"DEBUG: _capture_feature_importance called with model type: {type(model).__name__}")
        
        try:
            # 방법 1: PyCaret의 interpret_model 시도 (기본적으로 feature_importance 사용)
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            
            try:
                # matplotlib backend를 non-interactive로 설정하여 plot 창이 뜨지 않도록 함
                import matplotlib
                matplotlib.use('Agg')  # Non-interactive backend
                
                from pycaret.regression import interpret_model
                # PyCaret의 interpret_model을 사용하여 feature importance 추출
                interpret_model(model, plot='feature', save=False)
                
                # PyCaret 내부에서 feature importance 가져오기
                from pycaret.regression import get_config
                X_train = get_config('X_train')
                
                # 모델 타입에 따른 feature importance 추출
                if hasattr(model, 'feature_importances_'):
                    # Tree-based 모델 (RF, GBM, XGBoost 등)
                    importances = model.feature_importances_
                    feature_names = X_train.columns.tolist()
                    
                    for i, importance in enumerate(importances):
                        importance_list.append({
                            'feature': feature_names[i],
                            'importance': float(importance),
                            'rank': 0  # 나중에 정렬 후 랭크 부여
                        })
                        
                elif hasattr(model, 'coef_'):
                    # Linear 모델 (LR, Ridge, Lasso 등)
                    coefs = model.coef_
                    feature_names = X_train.columns.tolist()
                    
                    # 절대값으로 중요도 계산
                    for i, coef in enumerate(coefs):
                        importance_list.append({
                            'feature': feature_names[i],
                            'importance': abs(float(coef)),
                            'rank': 0
                        })
                
            except Exception as e1:
                # interpret_model often fails with PyCaret pipelines, this is expected
                print(f"DEBUG: Method 1 (interpret_model) failed: {str(e1)}")
                pass  # Silently continue to next method
                
                # 방법 2: PyCaret의 plot_model 시도
                try:
                    from pycaret.regression import plot_model
                    plot_model(model, plot='feature', save=False)
                    
                    # 여기서도 feature importance 추출 시도
                    from pycaret.regression import get_config
                    X_train = get_config('X_train')
                    
                    if hasattr(model, 'feature_importances_'):
                        importances = model.feature_importances_
                        feature_names = X_train.columns.tolist()
                        
                        for i, importance in enumerate(importances):
                            importance_list.append({
                                'feature': feature_names[i],
                                'importance': float(importance),
                                'rank': 0
                            })
                            
                except Exception as e2:
                    # plot_model also often fails with pipelines, expected
                    print(f"DEBUG: Method 2 (plot_model) failed: {str(e2)}")
                    pass
                    
                    # 방법 3: 직접 모델 속성 접근
                    try:
                        from pycaret.regression import get_config
                        X_train = get_config('X_train')
                        feature_names = X_train.columns.tolist()
                        
                        # Pipeline에서 실제 모델 추출
                        actual_model = model
                        if hasattr(model, 'steps'):
                            # Pipeline인 경우 - 마지막 단계가 실제 모델
                            actual_model = model.steps[-1][1] if model.steps else model
                            print(f"DEBUG: Extracted model from pipeline: {type(actual_model).__name__}")
                        
                        # 중첩된 Pipeline 처리
                        if hasattr(actual_model, 'steps'):
                            actual_model = actual_model.steps[-1][1] if actual_model.steps else actual_model
                            print(f"DEBUG: Extracted model from nested pipeline: {type(actual_model).__name__}")
                        
                        if hasattr(actual_model, 'feature_importances_'):
                            importances = actual_model.feature_importances_
                            print(f"DEBUG: Found feature_importances_ with {len(importances)} features")
                            for i, importance in enumerate(importances):
                                if i < len(feature_names):
                                    importance_list.append({
                                        'feature': feature_names[i],
                                        'importance': float(importance),
                                        'rank': 0
                                    })
                        elif hasattr(actual_model, 'coef_'):
                            coefs = actual_model.coef_
                            if len(coefs.shape) > 1:
                                coefs = coefs[0]
                            print(f"DEBUG: Found coef_ with {len(coefs)} coefficients")
                            for i, coef in enumerate(coefs):
                                if i < len(feature_names):
                                    importance_list.append({
                                        'feature': feature_names[i],
                                        'importance': abs(float(coef)),
                                        'rank': 0
                                    })
                                    
                    except Exception as e3:
                        # Direct access might fail too, continue to fallback
                        print(f"DEBUG: Method 3 (direct access) failed: {str(e3)}")
                        pass
                        
                        # 실제 모델 속성에서 가져올 수 없으면 빈 리스트 반환
                        print("WARNING: Could not extract feature importance from model")
                        pass
                            
        finally:
            sys.stdout = old_stdout
        
        # 중요도로 정렬하고 랭크 부여
        if importance_list:
            importance_list.sort(key=lambda x: x['importance'], reverse=True)
            for i, item in enumerate(importance_list):
                item['rank'] = i + 1
                
            logging.info(f"Captured {len(importance_list)} feature importances")
            
        return importance_list
    
    def get_feature_importance(self, target: str = None) -> List[Dict[str, Any]]:
        """저장된 feature importance 반환"""
        if target == 'wage_increase_bu_sbl' or target == 'baseup':
            return self.baseup_feature_importance or []
        elif target == 'wage_increase_mi_sbl' or target == 'performance':
            return self.performance_feature_importance or []
        else:
            return self.current_feature_importance or []

    def save_models(self):
        """학습된 모델을 파일로 저장"""
        try:
            # 저장 디렉토리 생성
            if not os.path.exists(self.model_dir):
                os.makedirs(self.model_dir)
                print(f"📁 Created model directory: {self.model_dir}")
            
            # Base-up 모델 저장
            if self.baseup_model is not None:
                print(f"   - Saving baseup feature importance: {len(self.baseup_feature_importance) if self.baseup_feature_importance else 0} features")
                with open(self.baseup_model_path, 'wb') as f:
                    pickle.dump({
                        'model': self.baseup_model,
                        'feature_importance': self.baseup_feature_importance,
                        'target': 'wage_increase_bu_sbl'
                    }, f)
                print(f"💾 Base-up model saved to {self.baseup_model_path}")
            
            # Performance 모델 저장
            if self.performance_model is not None:
                print(f"   - Saving performance feature importance: {len(self.performance_feature_importance) if self.performance_feature_importance else 0} features")
                with open(self.performance_model_path, 'wb') as f:
                    pickle.dump({
                        'model': self.performance_model,
                        'feature_importance': self.performance_feature_importance,
                        'target': 'wage_increase_mi_sbl'
                    }, f)
                print(f"💾 Performance model saved to {self.performance_model_path}")
            
            # Current model도 저장 (현재 활성 모델)
            if self.current_model is not None:
                current_model_path = os.path.join(self.model_dir, "current_model.pkl")
                with open(current_model_path, 'wb') as f:
                    pickle.dump({
                        'model': self.current_model,
                        'feature_importance': self.current_feature_importance,
                        'target': self.current_target
                    }, f)
                print(f"💾 Current model saved to {current_model_path}")
            
            return {
                "message": "Models saved successfully",
                "baseup_saved": self.baseup_model is not None,
                "performance_saved": self.performance_model is not None,
                "current_saved": self.current_model is not None
            }
            
        except Exception as e:
            print(f"❌ Error saving models: {str(e)}")
            return {
                "error": f"Failed to save models: {str(e)}"
            }
    
    def load_saved_models(self):
        """저장된 모델을 파일에서 로드"""
        try:
            models_loaded = []
            
            # Base-up 모델 로드
            if os.path.exists(self.baseup_model_path):
                with open(self.baseup_model_path, 'rb') as f:
                    data = pickle.load(f)
                    self.baseup_model = data['model']
                    self.baseup_feature_importance = data.get('feature_importance', [])
                    models_loaded.append('baseup')
                print(f"✅ Base-up model loaded from {self.baseup_model_path}")
                print(f"   - Feature importance: {len(self.baseup_feature_importance) if self.baseup_feature_importance else 0} features")
            
            # Performance 모델 로드
            if os.path.exists(self.performance_model_path):
                with open(self.performance_model_path, 'rb') as f:
                    data = pickle.load(f)
                    self.performance_model = data['model']
                    self.performance_feature_importance = data.get('feature_importance', [])
                    models_loaded.append('performance')
                print(f"✅ Performance model loaded from {self.performance_model_path}")
                print(f"   - Feature importance: {len(self.performance_feature_importance) if self.performance_feature_importance else 0} features")
            
            # Current model 로드
            current_model_path = os.path.join(self.model_dir, "current_model.pkl")
            if os.path.exists(current_model_path):
                with open(current_model_path, 'rb') as f:
                    data = pickle.load(f)
                    self.current_model = data['model']
                    self.current_feature_importance = data.get('feature_importance', [])
                    self.current_target = data.get('target')
                    models_loaded.append('current')
                print(f"✅ Current model loaded from {current_model_path}")
            
            # 로드된 모델이 있으면 설정 완료 플래그 설정
            if models_loaded:
                self.is_setup_complete = True
                self.is_model_trained_individually = True
                print(f"🚀 Successfully loaded {len(models_loaded)} model(s): {', '.join(models_loaded)}")
            else:
                print("ℹ️ No saved models found. Please train models first.")
            
            return {
                "message": f"Loaded {len(models_loaded)} model(s)",
                "models_loaded": models_loaded,
                "ready": len(models_loaded) > 0
            }
            
        except Exception as e:
            print(f"⚠️ Error loading models: {str(e)}")
            print("ℹ️ Models will need to be retrained.")
            return {
                "error": f"Failed to load models: {str(e)}",
                "models_loaded": [],
                "ready": False
            }

# 싱글톤 인스턴스
modeling_service = ModelingService()