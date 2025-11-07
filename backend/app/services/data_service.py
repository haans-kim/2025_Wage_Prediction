import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import os
from pathlib import Path
import hashlib
from datetime import datetime
import pickle
import logging
from app.core.config import settings, UPLOAD_DIR, DATA_DIR

class DataService:
    def __init__(self):
        self.upload_dir = Path(UPLOAD_DIR)
        self.upload_dir.mkdir(exist_ok=True)
        self.data_dir = Path(DATA_DIR)
        self.data_dir.mkdir(exist_ok=True)
        self.pickle_file = self.data_dir / "master_data.pkl"  # 마스터 데이터
        self.working_pickle_file = self.data_dir / "working_data.pkl"  # 작업용 데이터

        # 마스터 데이터와 작업 데이터 분리
        self.master_data: Optional[pd.DataFrame] = None  # 원본 마스터 데이터
        self.current_data: Optional[pd.DataFrame] = None  # 현재 작업 데이터
        self.data_info: Optional[Dict[str, Any]] = None
        self.column_mapping: Dict[str, str] = {}  # 영문 -> 한글 매핑
        self.reverse_column_mapping: Dict[str, str] = {}  # 한글 -> 영문 매핑

        # 시작시 기본 데이터 로드 시도
        self._load_default_data()

    def _clear_models_on_data_change(self):
        """데이터 변경 시 모델 자동 초기화"""
        try:
            from app.services.modeling_service import modeling_service
            result = modeling_service.clear_models()
            print("[AUTO-CLEAR] Models automatically cleared due to data change")
            return result
        except Exception as e:
            print(f"[WARNING] Could not auto-clear models: {e}")
            return None
    
    def _load_default_data(self) -> bool:
        """기본 데이터 로드 (pickle 파일에서)"""
        try:
            # 마스터 데이터 로드
            if self.pickle_file.exists():
                with open(self.pickle_file, 'rb') as f:
                    data_package = pickle.load(f)
                    self.master_data = data_package['data']
                    self.current_data = self.master_data.copy()  # 작업용 복사본
                    self.data_info = data_package['info']
                    self.column_mapping = data_package.get('column_mapping', {})
                    self.reverse_column_mapping = data_package.get('reverse_column_mapping', {})
                    if 'target_column' in data_package:
                        self.target_column = data_package['target_column']
                    if 'year_column' in data_package:
                        self.year_column = data_package['year_column']
                    logging.info(f"Loaded master data from pickle: {self.master_data.shape}")
                    return True
        except Exception as e:
            logging.warning(f"Failed to load default data from pickle: {e}")
        return False
    
    def _save_master_data_to_pickle(self) -> None:
        """마스터 데이터를 pickle 파일로 저장 (기존 파일 삭제 후 저장)"""
        try:
            if self.master_data is not None and self.data_info is not None:
                # 기존 마스터 데이터 파일 삭제
                if self.pickle_file.exists():
                    self.pickle_file.unlink()
                    logging.info(f"Removed old master data pickle file")

                data_package = {
                    'data': self.master_data,
                    'info': self.data_info,
                    'column_mapping': self.column_mapping,
                    'reverse_column_mapping': self.reverse_column_mapping,
                    'target_column': getattr(self, 'target_column', None),
                    'year_column': getattr(self, 'year_column', None),
                    'timestamp': datetime.now().isoformat()
                }
                with open(self.pickle_file, 'wb') as f:
                    pickle.dump(data_package, f)
                logging.info(f"Saved master data to pickle: {self.master_data.shape}")
        except Exception as e:
            logging.error(f"Failed to save master data to pickle: {e}")
    
    def _save_working_data_to_pickle(self) -> None:
        """작업 데이터를 pickle 파일로 저장 (기존 파일 삭제 후 저장)"""
        try:
            if self.current_data is not None:
                # 기존 작업 데이터 파일 삭제
                if self.working_pickle_file.exists():
                    self.working_pickle_file.unlink()
                    logging.info(f"Removed old working data pickle file")

                data_package = {
                    'data': self.current_data,
                    'is_augmented': len(self.current_data) != len(self.master_data) if self.master_data is not None else False,
                    'augmentation_info': getattr(self, 'last_augmentation_info', None),
                    'timestamp': datetime.now().isoformat()
                }
                with open(self.working_pickle_file, 'wb') as f:
                    pickle.dump(data_package, f)
                logging.info(f"Saved working data to pickle: {self.current_data.shape}")
        except Exception as e:
            logging.error(f"Failed to save working data to pickle: {e}")
    
    def get_default_data_status(self) -> Dict[str, Any]:
        """기본 데이터 상태 확인"""
        return {
            "has_master_data": self.master_data is not None,
            "has_working_data": self.current_data is not None,
            "master_pickle_exists": self.pickle_file.exists(),
            "working_pickle_exists": self.working_pickle_file.exists(),
            "master_data_shape": self.master_data.shape if self.master_data is not None else None,
            "working_data_shape": self.current_data.shape if self.current_data is not None else None,
            "is_augmented": (len(self.current_data) != len(self.master_data)) if (self.current_data is not None and self.master_data is not None) else False,
            "pickle_file_path": str(self.pickle_file)
        }
    
    def save_uploaded_file(self, file_content: bytes, filename: str) -> str:
        """업로드된 파일을 저장하고 경로 반환"""
        # 파일명에 타임스탬프 추가하여 중복 방지
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name, ext = os.path.splitext(filename)
        safe_filename = f"{name}_{timestamp}{ext}"
        
        file_path = self.upload_dir / safe_filename
        with open(file_path, "wb") as f:
            f.write(file_content)
        
        return str(file_path)
    
    def load_data_from_file(self, file_path: str) -> Dict[str, Any]:
        """파일에서 데이터 로드 및 분석"""
        try:
            # 파일 확장자에 따라 적절한 로더 사용
            if file_path.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file_path)
            elif file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
            
            # 데이터 전처리
            df = self._preprocess_data(df)
            
            # 최종 데이터 정리
            df = self._clean_data_for_storage(df)
            
            # 마스터 데이터로 저장
            self.master_data = df
            self.current_data = df.copy()  # 작업용 복사본
            
            # 데이터 정보 생성
            data_info = self._analyze_dataframe(df)
            self.data_info = data_info
            
            # 마스터 데이터 pickle 파일로 저장
            self._save_master_data_to_pickle()
            
            return data_info
            
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")
    
    def _analyze_dataframe(self, df: pd.DataFrame) -> Dict[str, Any]:
        """DataFrame 분석 및 메타데이터 생성"""
        # 기본 통계
        basic_stats = {
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "memory_usage": int(df.memory_usage(deep=True).sum()),  # numpy int를 Python int로 변환
        }
        
        # 결측값 분석 (numpy 타입을 Python 타입으로 변환)
        missing_counts = df.isnull().sum()
        missing_analysis = {
            "missing_counts": {k: int(v) for k, v in missing_counts.to_dict().items()},
            "missing_percentages": {k: float(v) for k, v in (missing_counts / len(df) * 100).round(2).to_dict().items()},
            "total_missing": int(missing_counts.sum()),
        }
        
        # 수치형 컬럼 통계
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_stats = {}
        if numeric_columns:
            describe_dict = df[numeric_columns].describe().to_dict()
            # numpy 타입을 Python 타입으로 변환
            numeric_stats = {
                col: {k: float(v) if not pd.isna(v) else None for k, v in stats.items()}
                for col, stats in describe_dict.items()
            }
        
        # 범주형 컬럼 분석
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        categorical_stats = {}
        for col in categorical_columns:
            value_counts = df[col].value_counts().head(5)
            categorical_stats[col] = {
                "unique_count": int(df[col].nunique()),
                "top_values": {k: int(v) for k, v in value_counts.to_dict().items()}
            }
        
        # 샘플 데이터 (전체 데이터 반환)
        sample_data = df.fillna("").to_dict(orient="records")
        
        return {
            "basic_stats": basic_stats,
            "missing_analysis": missing_analysis,
            "numeric_stats": numeric_stats,
            "categorical_stats": categorical_stats,
            "sample_data": sample_data,
            "numeric_columns": numeric_columns,
            "categorical_columns": categorical_columns,
        }
    
    def get_sample_data(self, n_rows: int = 100) -> Dict[str, Any]:
        """현재 데이터의 샘플 반환"""
        if self.current_data is None:
            raise ValueError("No data loaded")
        
        sample_df = self.current_data.head(n_rows)
        return {
            "data": sample_df.fillna("").to_dict(orient="records"),
            "shape": sample_df.shape,
            "columns": sample_df.columns.tolist()
        }
    
    def get_data_summary(self) -> Dict[str, Any]:
        """현재 데이터의 요약 정보 반환"""
        if self.data_info is None:
            raise ValueError("No data loaded")
        
        return {
            "shape": self.data_info["basic_stats"]["shape"],
            "columns": self.data_info["basic_stats"]["columns"],
            "numeric_columns": self.data_info["numeric_columns"],
            "categorical_columns": self.data_info["categorical_columns"],
            "missing_data_percentage": (
                self.data_info["missing_analysis"]["total_missing"] / 
                (self.data_info["basic_stats"]["shape"][0] * self.data_info["basic_stats"]["shape"][1]) * 100
            ),
            "memory_usage_mb": self.data_info["basic_stats"]["memory_usage"] / 1024 / 1024
        }
    
    def validate_data_for_modeling(self) -> Dict[str, Any]:
        """모델링을 위한 데이터 검증"""
        if self.current_data is None:
            raise ValueError("No data loaded")
        
        df = self.current_data
        issues = []
        
        # 최소 행 수 확인
        if len(df) < 10:
            issues.append("데이터가 너무 적습니다 (최소 10행 필요)")
        
        # 수치형 컬럼 확인
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            issues.append("수치형 컬럼이 부족합니다 (최소 2개 필요)")
        
        # 결측값 비율 확인
        missing_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns))
        if missing_ratio > 0.5:
            issues.append(f"결측값이 너무 많습니다 ({missing_ratio:.1%})")
        
        # 중복 행 확인
        duplicates = df.duplicated().sum()
        if duplicates > len(df) * 0.1:
            issues.append(f"중복 행이 많습니다 ({duplicates}개)")
        
        return {
            "is_valid": len(issues) == 0,
            "issues": issues,
            "recommendations": self._get_data_recommendations(df)
        }
    
    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """데이터 전처리: 한글/영문 Feature 처리, 연도 컬럼 제외, 타겟 설정"""
        # 컬럼명 정리
        columns = df.columns.tolist()
        
        # 첫 번째 행이 영문 Feature명인지 확인
        first_row = df.iloc[0] if len(df) > 0 else None
        if first_row is not None and all(isinstance(val, str) for val in first_row.values):
            # 첫 번째 행을 영문 컬럼명으로 사용
            english_columns = first_row.tolist()
            korean_columns = columns
            
            # 영문 컬럼명으로 변경
            df.columns = english_columns
            # 첫 번째 행 제거
            df = df.iloc[1:].reset_index(drop=True)
            
            # 컬럼 매핑 정보 저장 (양방향)
            self.column_mapping = dict(zip(english_columns, korean_columns))  # 영문 -> 한글
            self.reverse_column_mapping = dict(zip(korean_columns, english_columns))  # 한글 -> 영문
            logging.info(f"Column mapping created: {len(self.column_mapping)} columns")
        
        # 데이터 타입 변환 (숫자형으로 변환 가능한 컬럼은 변환)
        for col in df.columns:
            try:
                # 숫자형으로 변환
                df[col] = pd.to_numeric(df[col], errors='coerce')
                        
            except Exception as e:
                logging.warning(f"Error processing column {col}: {e}")
                pass
        
        # 타겟 컬럼 식별
        target_column = None
        wage_increase_keywords = ['임금인상율', '임금인상률', 'wage_increase', 'wage increase rate', 'salary increase']
        
        for col in df.columns:
            col_lower = str(col).lower()
            if any(keyword in col_lower for keyword in ['임금인상', 'wage_increase', 'salary_increase']):
                target_column = col
                self.target_column = col
                logging.info(f"Target column identified: {target_column}")
                break
        
        # 연도 컬럼 식별 및 메타데이터로 저장 (Feature에서는 제외)
        year_column = None
        year_keywords = ['year', 'Year', 'YEAR', '년도', '연도']
        
        for col in df.columns:
            if str(col) in year_keywords or str(col).lower() == 'year':
                year_column = col
                self.year_column = col
                logging.info(f"Year column identified: {year_column}")
                break
        
        # Base-up과 성과인상률 컬럼 식별
        baseup_column = None
        performance_column = None
        
        for col in df.columns:
            col_lower = str(col).lower()
            if 'base' in col_lower or 'base-up' in col_lower or 'baseup' in col_lower:
                baseup_column = col
                logging.info(f"Base-up column identified: {baseup_column}")
            elif 'performance' in col_lower or '성과' in col_lower:
                performance_column = col
                logging.info(f"Performance column identified: {performance_column}")
        
        return df
    
    def _clean_data_for_storage(self, df: pd.DataFrame) -> pd.DataFrame:
        """저장 전 데이터 최종 정리"""
        # 최소한의 정리만 수행, PyCaret이 나머지 처리
        return df
    
    def _get_data_recommendations(self, df: pd.DataFrame) -> list:
        """데이터 개선 권고사항"""
        recommendations = []
        
        # 결측값 처리 권고
        missing_cols = df.columns[df.isnull().any()].tolist()
        if missing_cols:
            recommendations.append(f"결측값 처리 필요: {', '.join(missing_cols[:3])}")
        
        # 이상치 검출 권고
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols[:3]:  # 처음 3개 컬럼만 확인
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
            if outliers > len(df) * 0.05:
                recommendations.append(f"{col} 컬럼에 이상치 검토 필요")
        
        return recommendations
    
    def get_model_config(self) -> Dict[str, Any]:
        """모델링을 위한 설정 정보 반환"""
        config = {
            "target_column": getattr(self, 'target_column', None),
            "year_column": getattr(self, 'year_column', None),
            "column_mapping": self.column_mapping,  # 영문 -> 한글
            "reverse_column_mapping": self.reverse_column_mapping,  # 한글 -> 영문
            "feature_columns": []
        }
        
        if self.current_data is not None:
            # Feature 컬럼 = 전체 컬럼 - 타겟 컬럼 - 연도 컬럼
            all_columns = self.current_data.columns.tolist()
            exclude_columns = []
            
            if config["target_column"]:
                exclude_columns.append(config["target_column"])
            if config["year_column"]:
                exclude_columns.append(config["year_column"])
            
            config["feature_columns"] = [col for col in all_columns if col not in exclude_columns]
        
        return config
    
    def get_korean_column_name(self, english_name: str) -> str:
        """영문 컬럼명을 한글 컬럼명으로 변환"""
        return self.column_mapping.get(english_name, english_name)
    
    def get_english_column_name(self, korean_name: str) -> str:
        """한글 컬럼명을 영문 컬럼명으로 변환"""
        return self.reverse_column_mapping.get(korean_name, korean_name)
    
    def get_display_names(self, column_list: list) -> Dict[str, str]:
        """컬럼 리스트에 대한 표시용 이름(한글) 반환"""
        display_names = {}
        for col in column_list:
            display_names[col] = self.get_korean_column_name(col)
        return display_names
    
    def reset_to_master(self) -> Dict[str, Any]:
        """작업 데이터를 마스터 데이터로 리셋"""
        if self.master_data is None:
            raise ValueError("No master data available")
        
        self.current_data = self.master_data.copy()
        self.last_augmentation_info = None
        
        return {
            "message": "Reset to master data",
            "master_shape": self.master_data.shape,
            "working_shape": self.current_data.shape
        }
    
    def augment_data_with_noise(self, target_size: int = 120, noise_factor: float = 0.01) -> Dict[str, Any]:
        """Target 컬럼을 제외하고 나머지 데이터에 대해서 10배수로 증강"""
        if self.current_data is None:
            raise ValueError("No data loaded for augmentation")
        
        original_df = self.current_data.copy()
        original_size = len(original_df)
        
        # Target 컬럼 식별 (대소문자 무시하고 target이 포함된 컬럼 찾기)
        target_columns = [col for col in original_df.columns if 'target' in col.lower()]
        
        print(f"[DATA] Data Augmentation (Target 제외 10배수):")
        print(f"   - Original size: {original_size}")
        print(f"   - Target columns found: {target_columns}")
        print(f"   - Multiplier: 10")
        print(f"   - Expected result: {original_size * 10}")
        
        augmented_rows = []
        
        # 각 원본 행에 대해 10배 증강 (Target 컬럼 제외)
        for _, row in original_df.iterrows():
            # 원본 행 추가
            augmented_rows.append(row.to_dict())
            
            # 9번 복제하면서 노이즈 추가 (10배이므로 원본 1 + 복제 9 = 10)
            for i in range(9):
                new_row = row.to_dict()
                
                # Target 컬럼과 year 컬럼을 제외한 수치형 특성에만 노이즈 추가
                for col in original_df.columns:
                    # Target 컬럼과 year 컬럼은 제외
                    if (col not in target_columns and 
                        col != 'year' and 
                        pd.api.types.is_numeric_dtype(original_df[col])):
                        if pd.notna(new_row[col]) and new_row[col] != 0:
                            # ±1% 범위의 노이즈
                            noise = np.random.normal(0, abs(new_row[col]) * noise_factor)
                            new_row[col] = new_row[col] + noise
                
                augmented_rows.append(new_row)
        
        # 증강된 데이터프레임 생성
        augmented_df = pd.DataFrame(augmented_rows)
        
        # 년도 컬럼이 있는 경우 정렬
        year_columns = ['year', 'Year', 'YEAR', '년도', '연도']
        year_col = None
        for col in year_columns:
            if col in augmented_df.columns:
                year_col = col
                break
        
        if year_col:
            augmented_df = augmented_df.sort_values(year_col).reset_index(drop=True)
        
        # 증강된 데이터로 업데이트
        self.current_data = augmented_df
        self.data_info = self._analyze_dataframe(augmented_df)

        # pickle 파일로 저장
        self._save_data_to_pickle()

        # 데이터 변경으로 인한 모델 자동 초기화
        self._clear_models_on_data_change()

        actual_size = len(augmented_df)
        augmented_rows_count = actual_size - original_size
        
        return {
            "message": f"Data augmented from {original_size} to {actual_size} rows (Target column excluded)",
            "original_size": original_size,
            "augmented_size": actual_size,
            "augmentation_applied": True,
            "multiplier": 10,
            "noise_factor": noise_factor,
            "augmented_rows": augmented_rows_count,
            "target_columns_excluded": target_columns,
            "method": "10x augmentation excluding Target columns"
        }

# 싱글톤 인스턴스
data_service = DataService()