import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import warnings
import io
import sys
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
        self.model_results = None
        self.compared_models = None  # 비교된 모델들
        self.is_setup_complete = False
        self.feature_names = None  # Store feature names for prediction
        self.prediction_data = None  # 2025년 예측 대상 데이터
        self.prediction_features = []  # 예측에 사용할 feature 컬럼명

        # 데이터 크기별 모델 리스트 정의
        self.small_data_models = ['ridge', 'lasso', 'lr', 'gbr']  # 작은 데이터용 모델
        self.medium_data_models = ['ridge', 'lasso', 'lr', 'gbr', 'rf', 'et']  # 중간 데이터용 모델
        self.large_data_models = ['ridge', 'lasso', 'lr', 'gbr', 'rf', 'et', 'xgboost', 'lightgbm']  # 큰 데이터용 모델

        # 하이브리드 모델 구성 (적은 데이터에 최적화)
        self.validation_models = {
            'ridge': None,  # 메인 예측 모델 (정규화로 과적합 방지)
            'lasso': None,  # Feature selection
            'lr': None,     # Baseline
            'gbr': None     # Feature importance & 비선형 패턴
        }

        # 모델 역할 정의
        self.model_roles = {
            'ridge': {'weight': 0.5, 'role': 'main_prediction', 'description': '안정적 선형 예측'},
            'lasso': {'weight': 0.2, 'role': 'feature_selection', 'description': '중요 변수 선택'},
            'lr': {'weight': 0.1, 'role': 'baseline', 'description': '기준선 모델'},
            'gbr': {'weight': 0.2, 'role': 'nonlinear_capture', 'description': '비선형 패턴 포착'}
        }

        # Feature importance 저장
        self.feature_importance = {}

        # 초기화 시 저장된 최신 모델 자동 로드 시도
        self._load_latest_model_if_exists()
    
    def check_pycaret_availability(self) -> bool:
        """PyCaret 사용 가능 여부 확인"""
        return PYCARET_AVAILABLE
    
    def get_optimal_settings(self, data_size: int) -> Dict[str, Any]:
        """데이터 크기에 따른 최적 설정 반환"""
        if data_size < 30:
            return {
                'train_size': 0.9,
                'cv_folds': 2 if data_size < 15 else 3,  # 매우 작은 데이터는 2-fold
                'models': self.small_data_models,
                'normalize': True,  # 작은 데이터도 정규화 적용
                'transformation': True,  # 작은 데이터도 변환 적용
                'remove_outliers': True,  # 작은 데이터도 이상치 제거
                'feature_selection': False,  # GBR 사용으로 특성 선택 불필요
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
                'feature_selection': False,  # GBR 사용으로 특성 선택 불필요
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
                'feature_selection': False,  # GBR 사용으로 특성 선택 불필요
                'n_features_to_select': 0.6
            }
    
    def prepare_data_for_modeling(self, target_column: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """모델링을 위한 데이터 준비"""
        if data_service.current_data is None:
            raise ValueError("No data loaded for modeling")
        
        df = data_service.current_data.copy()
        
        # data_service에서 설정된 컬럼 정보 가져오기
        model_config = data_service.get_model_config()
        
        # 타겟 컬럼 결정 (인자로 받은 것 우선, 없으면 자동 감지된 것 사용)
        if target_column is None:
            target_column = model_config.get('target_column')
            if target_column is None:
                # 마지막 컬럼을 타겟으로 가정
                target_column = df.columns[-1]
                logging.info(f"No target column specified, using last column: {target_column}")
        
        # 기본 데이터 정리
        info = {
            'original_shape': df.shape,
            'target_column': target_column,
            'numeric_columns': [],
            'categorical_columns': [],
            'dropped_columns': [],
            'year_column': model_config.get('year_column'),
            'feature_columns': model_config.get('feature_columns', [])
        }
        
        # 타겟 컬럼 존재 확인
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")
        
        # 2025년 데이터(예측 대상) 분리 저장
        prediction_data_mask = df[target_column].isna()
        if prediction_data_mask.any():
            self.prediction_data = df[prediction_data_mask].copy()
            self.prediction_features = model_config.get('feature_columns', [])
            print(f"[DATA] Separated {len(self.prediction_data)} rows for 2026 prediction (2025 data with no target)")
        else:
            self.prediction_data = None
            self.prediction_features = []
        
        # 타겟 컬럼에 결측값이 있는 행 제거 (2025년 예측 대상 데이터 제외)
        initial_rows = len(df)
        df = df.dropna(subset=[target_column])
        removed_rows = initial_rows - len(df)
        
        if removed_rows > 0:
            info['removed_target_missing'] = removed_rows
            print(f"[DATA] Removed {removed_rows} rows with missing target values (likely future prediction data)")
        
        # 타겟 컬럼이 충분한 데이터가 있는지 확인
        if len(df) < 5:
            raise ValueError(f"Insufficient training data: only {len(df)} rows with valid target values")
        
        # 최소한의 전처리만 수행 (PyCaret이 나머지를 처리)
        # '-' 값을 NaN으로 변환 (PyCaret이 인식할 수 있도록)
        df = df.replace(['-', ''], np.nan)
        
        # 데이터 누수 방지: 임금 관련 컬럼 제거 (타겟과 직접 관련)
        wage_columns_to_remove = [
            'wage_increase_bu_sbl',     # Base-up (타겟의 일부)
            'wage_increase_mi_sbl',      # 성과급 (타겟의 일부)
            'wage_increase_baseup_sbl',  # Base-up 다른 이름
            'Base-up 인상률',            # 한글명
            '성과인상률',                # 한글명
        ]
        
        for col in wage_columns_to_remove:
            if col in df.columns and col != target_column:
                df = df.drop(columns=[col])
                info['dropped_columns'].append(col)
                print(f"[DATA] Removed wage-related column (data leakage prevention): {col}")
        
        # 연도 컬럼 제거 (data_service에서 식별된 것 사용)
        if info['year_column'] and info['year_column'] in df.columns:
            if info['year_column'] != target_column:
                df = df.drop(columns=[info['year_column']])
                info['dropped_columns'].append(info['year_column'])
                print(f"[DATA] Removed year column: {info['year_column']}")
        else:
            # 백업: 수동으로 연도 컬럼 찾기
            year_columns = ['year', 'Year', 'YEAR', '년도', '연도', 'eng', 'kor']
            for year_col in year_columns:
                if year_col in df.columns and year_col != target_column:
                    df = df.drop(columns=[year_col])
                    info['dropped_columns'].append(year_col)
                    print(f"[DATA] Removed year column: {year_col}")
        
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
        target_column: Optional[str] = None, 
        train_size: Optional[float] = None,
        session_id: int = 42,  # 고정된 시드값 사용
        preprocessing_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """PyCaret 환경 설정 (전처리 옵션 포함)"""
        
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
            # 데이터 체크 디버깅
            print(f"[DATA] Before setup - Data shape: {ml_data.shape}")
            print(f"[DATA] Data types: {ml_data.dtypes.value_counts()}")
            
            # 문제가 될 수 있는 값 체크 및 수정
            for col in ml_data.columns:
                if pd.api.types.is_numeric_dtype(ml_data[col]):
                    # Infinity 값 처리
                    if ml_data[col].isin([np.inf, -np.inf]).any():
                        print(f"[WARNING] Column {col} contains infinity - replacing with NaN")
                        ml_data[col] = ml_data[col].replace([np.inf, -np.inf], np.nan)
                    
                    # 매우 큰 값 스케일링 (백만 단위로 변환)
                    max_val = ml_data[col].max()
                    if pd.notna(max_val) and abs(max_val) > 1e7:
                        print(f"[DATA] Column {col} has large values (max: {max_val:.2e}) - scaling down")
                        # 백만 단위로 스케일링
                        ml_data[col] = ml_data[col] / 1e6
                        print(f"  → Scaled to max: {ml_data[col].max():.2f}M")
            
            # 모든 출력 억제
            sys.stdout = io.StringIO()
            sys.stderr = io.StringIO()
            
            # 전처리 설정 병합
            if preprocessing_config:
                # 사용자 정의 설정이 있으면 우선 사용
                config = preprocessing_config
            else:
                # 기본 설정 사용
                config = {
                    'imputation_type': 'simple',
                    'numeric_imputation': 'mean',
                    'categorical_imputation': 'mode',
                    'normalize': optimal_settings['normalize'],
                    'normalize_method': 'zscore' if optimal_settings['normalize'] else None,
                    'transformation': optimal_settings['transformation'],
                    'transformation_method': 'yeo-johnson' if optimal_settings['transformation'] else None,
                    'remove_outliers': optimal_settings['remove_outliers'],
                    'outliers_threshold': 0.05 if optimal_settings['remove_outliers'] else None,
                    'remove_multicollinearity': True,
                    'multicollinearity_threshold': 0.9,
                    'feature_selection': optimal_settings['feature_selection']
                }
            
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
                imputation_type=config.get('imputation_type', 'simple'),
                numeric_imputation=config.get('numeric_imputation', 'mean'),
                categorical_imputation=config.get('categorical_imputation', 'mode'),
                
                # 정규화
                normalize=config.get('normalize', True),
                normalize_method=config.get('normalize_method', 'zscore'),
                
                # 변환
                transformation=config.get('transformation', False),
                transformation_method=config.get('transformation_method', 'yeo-johnson'),
                
                # 이상치 제거
                remove_outliers=config.get('remove_outliers', False),
                outliers_threshold=config.get('outliers_threshold', 0.05),
                
                # 다중공선성 제거
                remove_multicollinearity=config.get('remove_multicollinearity', True),
                multicollinearity_threshold=config.get('multicollinearity_threshold', 0.9),
                
                # 특성 선택 비활성화 (GBR 모델 사용으로 불필요)
                feature_selection=False,
                
                # CV 전략
                fold_strategy='kfold',
                fold=optimal_settings['cv_folds']
            )
            
            self.current_experiment = exp
            self.is_setup_complete = True
            
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
        
        if not self.is_setup_complete and self.current_model is None:
            raise RuntimeError("PyCaret environment not setup. Call setup_pycaret_environment first.")
        
        # 현재 데이터 크기 확인
        data_size = len(data_service.current_data)
        optimal_settings = self.get_optimal_settings(data_size)
        models_to_use = optimal_settings['models']
        
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        
        try:
            # 출력 억제
            sys.stdout = io.StringIO()
            sys.stderr = io.StringIO()
            
            # 모델 비교 실행하되 GBR이 최상위에 오도록 조정
            best_models = compare_models(
                include=models_to_use,
                sort='R2',
                n_select=min(n_select, len(models_to_use)),
                verbose=False,
                fold=3  # 빠른 비교를 위해 fold 수 제한
            )
            
            # 단일 모델이 반환된 경우 리스트로 변환
            if not isinstance(best_models, list):
                best_models = [best_models]
            
            # 결과 정보 추출
            comparison_results = pull()
            
            # GBR이 포함되어 있으면 최상위로 이동, 없으면 추가
            gbr_model = None
            for model in best_models:
                if hasattr(model, '__class__') and 'GradientBoosting' in str(type(model)):
                    gbr_model = model
                    break
            
            # GBR이 없으면 새로 생성해서 최상위에 추가
            if gbr_model is None:
                print("🔧 GBR not in best models, creating and adding to top")
                gbr_model = create_model('gbr', verbose=False, random_state=42)
                best_models = [gbr_model] + best_models
            else:
                # GBR이 있으면 최상위로 이동
                print("🔧 Moving GBR to top of best models")
                best_models.remove(gbr_model)
                best_models = [gbr_model] + best_models
            
            # feature names 저장
            from pycaret.regression import get_config
            X_train = get_config('X_train')
            if X_train is not None:
                self.feature_names = list(X_train.columns)
                print(f"[DATA] Stored feature names: {len(self.feature_names)} features")
            
            self.model_results = {
                'best_models': best_models,
                'comparison_df': comparison_results,
                'recommended_model': best_models[0],  # 첫 번째는 항상 GBR
                'gbr_prioritized': True  # GBR 우선 선정 표시
            }
            # 모델 비교 후에는 current_model을 설정하지 않음 (명시적 학습 필요)
            # self.current_model = best_models[0] if best_models else None
            self.compared_models = best_models  # 비교된 모델들만 저장
            
        except Exception as e:
            # 실패 시 기본 GBR 사용 (더 나은 성능을 위해)
            warnings.warn(f"Model comparison failed: {str(e)}. Using default Gradient Boosting Regressor.")
            
            gbr_model = create_model('gbr', verbose=False, random_state=42)
            self.model_results = {
                'best_models': [gbr_model],
                'comparison_df': None,
                'recommended_model': gbr_model,
                'fallback_used': True
            }
            
        finally:
            # 출력 복원
            sys.stdout = old_stdout
            sys.stderr = old_stderr
        
        return {
            'message': 'Model comparison completed',
            'models_compared': len(models_to_use),
            'best_model_count': len(self.model_results['best_models']),
            'recommended_model_type': type(self.model_results['recommended_model']).__name__,
            'comparison_available': self.model_results['comparison_df'] is not None,
            'data_size_category': 'small' if data_size < 30 else 'medium' if data_size < 100 else 'large'
        }
    
    def train_specific_model(self, model_name: str) -> Dict[str, Any]:
        """특정 모델 학습"""
        
        if not self.is_setup_complete and self.current_model is None:
            raise RuntimeError("PyCaret environment not setup. Call setup_pycaret_environment first.")
        
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        
        try:
            # 출력 억제
            sys.stdout = io.StringIO()
            sys.stderr = io.StringIO()
            
            # 모델 생성 (모델별로 random_state 지원 여부 확인)
            # Linear models (lr, ridge, lasso 등)은 random_state를 지원하지 않음
            models_without_random_state = ['lr', 'ridge', 'lasso', 'en', 'lar', 'llar', 'omp', 'br', 'ard', 'par', 'ransac', 'tr', 'huber']
            
            if model_name in models_without_random_state:
                model = create_model(model_name, verbose=False)
            else:
                model = create_model(model_name, verbose=False, random_state=42)
            
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
            
            # 모델 자동 저장
            self._save_model(model_name)
            
        except Exception as e:
            raise RuntimeError(f"Model training failed: {str(e)}")
        finally:
            # 출력 복원
            sys.stdout = old_stdout
            sys.stderr = old_stderr
        
        return {
            'message': f'Model {model_name} trained and saved successfully',
            'model_type': type(self.current_model).__name__,
            'model_name': model_name,
            'model_saved': True
        }
    
    def get_model_evaluation(self) -> Dict[str, Any]:
        """현재 모델의 평가 결과 반환"""
        
        if self.current_model is None:
            # 모델 비교만 하고 학습하지 않은 경우 에러 반환
            raise RuntimeError("No trained model available. Please train a model first after comparison.")
        
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
                sys.stdout = io.StringIO()
                
                predictions = predict_model(self.current_model, data=prediction_data)
                
            except Exception as e:
                raise RuntimeError(f"Prediction with custom data failed: {str(e)}")
            finally:
                sys.stdout = old_stdout
            
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
            'model_trained': self.current_model is not None,
            'models_compared': self.model_results is not None,
            'data_loaded': data_service.current_data is not None,
            'current_model_type': type(self.current_model).__name__ if self.current_model else None
        }
    
    def _save_model(self, model_name: str = None) -> bool:
        """모델을 파일로 저장"""
        try:
            if self.current_model is None:
                return False
            
            # 저장 경로 설정
            import os
            import glob
            from datetime import datetime

            # models 디렉토리 생성
            models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models')
            os.makedirs(models_dir, exist_ok=True)

            # 기존 모델 파일들 모두 삭제 (latest.pkl 포함)
            old_model_files = glob.glob(os.path.join(models_dir, '*.pkl'))
            for old_file in old_model_files:
                try:
                    os.remove(old_file)
                    print(f"[CLEANUP] Removed old model file: {os.path.basename(old_file)}")
                except Exception as e:
                    print(f"[WARNING] Could not remove {os.path.basename(old_file)}: {e}")

            # 파일명 생성 (모델명_날짜시간.pkl)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            if model_name:
                filename = f"wage_model_{model_name}_{timestamp}"
            else:
                filename = f"wage_model_{timestamp}"

            filepath = os.path.join(models_dir, filename)

            # PyCaret의 save_model 사용
            from pycaret.regression import save_model
            save_model(self.current_model, filepath, verbose=False)

            # 최신 모델 링크 생성 (latest.pkl)
            latest_path = os.path.join(models_dir, 'latest')
            save_model(self.current_model, latest_path, verbose=False)

            print(f"[OK] Model saved successfully: {filename}.pkl (old models removed)")
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to save model: {str(e)}")
            return False
    
    def _load_latest_model_if_exists(self) -> bool:
        """초기화 시 최신 모델 자동 로드"""
        try:
            import os
            
            models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models')
            latest_path = os.path.join(models_dir, 'latest.pkl')
            
            # 최신 모델 파일이 존재하는지 확인
            if os.path.exists(latest_path):
                from pycaret.regression import load_model
                self.current_model = load_model(os.path.join(models_dir, 'latest'))
                print(f"[OK] Latest model loaded automatically from {latest_path}")
                return True
            else:
                print("[INFO] No saved model found. Will create new model when training.")
                return False

        except Exception as e:
            print(f"[WARNING] Could not load saved model: {str(e)}")
            return False
    
    def load_saved_model(self, filename: str = 'latest') -> Dict[str, Any]:
        """저장된 모델 불러오기"""
        try:
            import os
            from pycaret.regression import load_model
            
            models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models')
            filepath = os.path.join(models_dir, filename)
            
            # 모델 로드
            self.current_model = load_model(filepath)
            
            return {
                'message': f'Model loaded successfully from {filename}.pkl',
                'model_type': type(self.current_model).__name__
            }
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")
    
    def clear_models(self) -> Dict[str, Any]:
        """모든 모델 및 실험 초기화"""
        self.current_experiment = None
        self.current_model = None
        self.model_results = None
        self.is_setup_complete = False
        
        return {
            'message': 'All models and experiments cleared successfully'
        }

# 싱글톤 인스턴스
modeling_service = ModelingService()