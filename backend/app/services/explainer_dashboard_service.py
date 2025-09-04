"""
ExplainerDashboard 서비스
"""
import os
import pickle
import logging
import threading
import time
from typing import Optional, Dict, Any
from pathlib import Path

import pandas as pd
import numpy as np
from explainerdashboard import ClassifierExplainer, RegressionExplainer, ExplainerDashboard
from app.services.data_service import data_service

logger = logging.getLogger(__name__)


class ExplainerDashboardService:
    """ExplainerDashboard 생성 및 관리"""
    
    def __init__(self):
        self.dashboard: Optional[ExplainerDashboard] = None
        self.dashboard_thread: Optional[threading.Thread] = None
        self.dashboard_port: int = 8050
        self.is_running: bool = False
        self.dashboard_url: Optional[str] = None
        
    def create_dashboard(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series, 
                        feature_names: list, model_name: str = "Wage Increase Model") -> Dict[str, Any]:
        """ExplainerDashboard 생성"""
        try:
            logger.info("Creating ExplainerDashboard...")
            logger.info(f"X_test shape: {X_test.shape}, features: {list(X_test.columns)[:5]}...")
            logger.info(f"feature_names: {feature_names[:5]}...")
            
            # 모델이 이미 있으면 종료
            if self.is_running:
                self.stop_dashboard()
            
            # 모델이 훈련된 실제 feature 이름들 가져오기
            if hasattr(model, 'feature_names_in_'):
                model_features = list(model.feature_names_in_)
                logger.info(f"Model was trained with features: {model_features[:5]}...")
            else:
                # PyCaret 모델의 경우 다른 방법 시도
                try:
                    from pycaret.regression import get_config
                    model_features = list(get_config('X_train').columns)
                    logger.info(f"PyCaret model features: {model_features[:5]}...")
                except:
                    model_features = feature_names
                    logger.warning("Could not determine model features, using provided feature_names")
            
            # X_test를 모델이 훈련된 feature만 포함하도록 필터링
            if hasattr(X_test, 'columns'):
                actual_features = list(X_test.columns)
                
                # 모델이 훈련된 feature만 선택
                common_features = [f for f in model_features if f in actual_features]
                if len(common_features) != len(model_features):
                    logger.warning(f"Feature mismatch: model expects {len(model_features)}, "
                                 f"but X_test has {len(common_features)} matching features")
                    logger.info(f"Model features: {model_features}")
                    logger.info(f"X_test features: {actual_features}")
                    logger.info(f"Common features: {common_features}")
                
                # X_test를 모델 feature에 맞게 필터링
                X_test = X_test[common_features].copy()
                feature_names = common_features
                logger.info(f"Filtered X_test to shape: {X_test.shape} with features: {feature_names[:5]}...")
            
            # data_service에서 한글 컬럼명 가져오기
            feature_descriptions = data_service.get_display_names(feature_names)
            
            # 한글 → 영문 매핑 생성
            korean_to_english = {}
            english_to_korean = {}
            korean_feature_names = []
            for feat in feature_names:
                korean_name = feature_descriptions.get(feat, feat)
                korean_feature_names.append(korean_name)
                korean_to_english[korean_name] = feat
                english_to_korean[feat] = korean_name
            
            # 모델 래퍼 클래스 정의 (한글 컬럼명 → 영문 컬럼명 변환)
            class ModelWrapper:
                def __init__(self, original_model, korean_to_english):
                    self.model = original_model
                    self.korean_to_english = korean_to_english
                    
                def predict(self, X):
                    # 한글 컬럼명을 영문으로 변환
                    if isinstance(X, pd.DataFrame):
                        X_english = X.rename(columns=self.korean_to_english)
                    else:
                        # numpy array인 경우 DataFrame으로 변환
                        X_english = pd.DataFrame(X, columns=list(self.korean_to_english.keys()))
                        X_english = X_english.rename(columns=self.korean_to_english)
                    return self.model.predict(X_english)
                
                def __getattr__(self, name):
                    # 다른 속성들은 원본 모델로 전달
                    return getattr(self.model, name)
            
            # 래핑된 모델 생성
            wrapped_model = ModelWrapper(model, korean_to_english)
            
            # X_test에 한글 feature names 설정
            if isinstance(X_test, pd.DataFrame):
                X_test.columns = korean_feature_names
            else:
                X_test = pd.DataFrame(X_test, columns=korean_feature_names)
            
            # 원본 X_test 데이터 복사 (원본 수정 방지)
            X_test_copy = X_test.copy()
            
            # 한글 컬럼명 적용 (복사본에)
            new_columns = []
            for col in X_test_copy.columns:
                korean_name = feature_descriptions.get(col, col)
                new_columns.append(korean_name)
            X_test_copy.columns = new_columns
            
            # 원본 데이터만 표시하도록 인덱스 설정
            # 데이터 크기를 확인하여 원본만 선택
            if len(X_test_copy) > 20:  # 증강된 데이터가 있는 경우
                # 10개씩 묶여있다고 가정하고 첫 번째만 원본
                original_indices = []
                for i in range(len(X_test_copy)):
                    if i % 10 == 0:
                        original_indices.append(i)
                
                if original_indices:
                    X_test_copy = X_test_copy.iloc[original_indices]
                    y_test = y_test.iloc[original_indices] if y_test is not None else None
                    
            # y_test 형태 및 타입 확인 및 수정
            if y_test is not None:
                logger.info(f"Original y_test type: {type(y_test)}, shape: {getattr(y_test, 'shape', 'no shape')}")
                
                # numpy array나 scalar인 경우 pandas Series로 변환
                if isinstance(y_test, (int, float, np.number)):
                    # 단일 스칼라값인 경우
                    logger.info("Converting scalar y_test to Series")
                    y_test = pd.Series([float(y_test)])
                elif isinstance(y_test, np.ndarray):
                    if y_test.ndim == 0:  # 0차원 배열 (스칼라)
                        logger.info("Converting 0-dim array y_test to Series")
                        y_test = pd.Series([float(y_test.item())])
                    elif y_test.ndim == 1 and len(y_test) == 1:  # 길이 1인 1차원 배열
                        logger.info("Converting single-element array y_test to Series")
                        y_test = pd.Series([float(y_test[0])])
                    else:
                        logger.info("Converting numpy array y_test to Series")
                        y_test = pd.Series(y_test.flatten())
                
                # pandas Series가 아닌 경우 변환
                if not isinstance(y_test, pd.Series):
                    logger.info(f"Converting {type(y_test)} y_test to Series")
                    if hasattr(y_test, '__len__') and len(y_test) > 0:
                        y_test = pd.Series(list(y_test))
                    else:
                        # 빈 데이터이거나 길이를 알 수 없는 경우 기본값 사용
                        y_test = pd.Series([0.05])  # 5% 기본 인상률
                
                # X_test와 길이 맞추기
                if len(y_test) != len(X_test_copy):
                    if len(y_test) == 1:
                        # y_test가 단일값이면 X_test 길이에 맞게 복제
                        logger.info(f"Replicating single y_test value to match X_test length: {len(X_test_copy)}")
                        y_test = pd.Series([y_test.iloc[0]] * len(X_test_copy))
                    else:
                        # 길이가 다르면 최소 길이로 맞춤
                        min_len = min(len(y_test), len(X_test_copy))
                        logger.info(f"Truncating to minimum length: {min_len}")
                        y_test = y_test.iloc[:min_len]
                        X_test_copy = X_test_copy.iloc[:min_len]
                
                logger.info(f"Final y_test type: {type(y_test)}, length: {len(y_test)}, values: {y_test.values[:3] if len(y_test) > 0 else 'empty'}")
            else:
                # y_test가 None인 경우 더미 데이터 생성
                logger.warning("y_test is None, creating dummy target values")
                y_test = pd.Series([0.05] * len(X_test_copy))  # 5% 기본 인상률
            
            # 인덱스를 연도로 설정
            num_samples = len(X_test_copy)
            if num_samples <= 10:
                # 원본 데이터만 있는 경우
                start_year = 2016  # 실제 데이터 시작 연도
                years = [f"{start_year + i}년" for i in range(num_samples)]
            else:
                # 여전히 많은 데이터가 있는 경우
                years = [f"데이터_{i+1}" for i in range(num_samples)]
            
            X_test_copy.index = years
            y_test.index = years
            
            # Explainer 생성 (회귀 모델) - 기본 파라미터만 사용
            explainer = RegressionExplainer(
                wrapped_model,  # 래핑된 모델 사용 
                X_test_copy,  # 복사본 사용
                y_test,
                units='%'  # 단위 설정
            )
            
            # 대시보드 생성 - 기본 설정만 사용
            self.dashboard = ExplainerDashboard(
                explainer,
                title="임금인상률 예측 모델 분석",
                description="2026년 임금인상률 예측 모델의 상세 분석 대시보드",
                port=self.dashboard_port,
                mode='dash'
            )
            
            # 별도 스레드에서 대시보드 실행
            self.dashboard_thread = threading.Thread(
                target=self._run_dashboard,
                daemon=True
            )
            self.dashboard_thread.start()
            
            # 대시보드가 시작될 때까지 대기
            time.sleep(3)
            
            self.is_running = True
            self.dashboard_url = f"http://localhost:{self.dashboard_port}"
            
            logger.info(f"ExplainerDashboard started at {self.dashboard_url}")
            
            return {
                "success": True,
                "url": self.dashboard_url,
                "port": self.dashboard_port,
                "message": "ExplainerDashboard가 성공적으로 생성되었습니다."
            }
            
        except Exception as e:
            logger.error(f"Failed to create ExplainerDashboard: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "message": "ExplainerDashboard 생성 중 오류가 발생했습니다."
            }
    
    def _run_dashboard(self):
        """대시보드 실행 (별도 스레드)"""
        try:
            self.dashboard.run(use_waitress=True)
        except Exception as e:
            logger.error(f"Dashboard runtime error: {str(e)}")
            self.is_running = False
    
    def stop_dashboard(self):
        """대시보드 중지"""
        try:
            if self.dashboard:
                # Dash 서버 종료
                if hasattr(self.dashboard, 'app') and hasattr(self.dashboard.app, 'server'):
                    func = self.dashboard.app.server.shutdown
                    func()
                
                self.dashboard = None
                self.is_running = False
                self.dashboard_url = None
                
                logger.info("ExplainerDashboard stopped")
                
        except Exception as e:
            logger.error(f"Failed to stop dashboard: {str(e)}")
    
    def get_status(self) -> Dict[str, Any]:
        """대시보드 상태 확인"""
        return {
            "is_running": self.is_running,
            "url": self.dashboard_url,
            "port": self.dashboard_port if self.is_running else None
        }
    
    def save_explainer(self, filepath: str):
        """Explainer 저장"""
        try:
            if self.dashboard and hasattr(self.dashboard, 'explainer'):
                self.dashboard.explainer.dump(filepath)
                logger.info(f"Explainer saved to {filepath}")
                return True
        except Exception as e:
            logger.error(f"Failed to save explainer: {str(e)}")
        return False
    
    def load_explainer(self, filepath: str) -> bool:
        """저장된 Explainer 로드"""
        try:
            from explainerdashboard import ExplainerDashboard
            
            # 저장된 explainer 로드
            dashboard = ExplainerDashboard.from_file(filepath)
            
            # 대시보드 실행
            self.dashboard = dashboard
            self.dashboard_thread = threading.Thread(
                target=self._run_dashboard,
                daemon=True
            )
            self.dashboard_thread.start()
            
            time.sleep(3)
            
            self.is_running = True
            self.dashboard_url = f"http://localhost:{self.dashboard_port}"
            
            logger.info(f"Explainer loaded and dashboard started at {self.dashboard_url}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load explainer: {str(e)}")
            return False


# 싱글톤 인스턴스
explainer_dashboard_service = ExplainerDashboardService()