import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import os
from pathlib import Path
import hashlib
from datetime import datetime
import pickle
import logging

class DataService:
    def __init__(self):
        self.upload_dir = Path("uploads")
        self.upload_dir.mkdir(exist_ok=True)
        self.data_dir = Path("data")
        self.data_dir.mkdir(exist_ok=True)
        self.pickle_file = self.data_dir / "master_data.pkl"
        # 원본 데이터를 우선 사용 (문자열 값 처리를 PyCaret에 맡김)
        self.master_excel_file = "../SambaWage_250825.xlsx"
        self.current_data: Optional[pd.DataFrame] = None
        self.data_info: Optional[Dict[str, Any]] = None
        
        # 시작시 마스터 데이터 로드 시도
        self._load_master_data()
    
    def _load_master_data(self) -> bool:
        """마스터 데이터 로드 (pickle 파일 우선, 없으면 Excel에서)"""
        try:
            # 1. 먼저 pickle 파일에서 로드 시도
            if self.pickle_file.exists():
                with open(self.pickle_file, 'rb') as f:
                    data_package = pickle.load(f)
                    self.current_data = data_package['data']
                    self.data_info = data_package['info']
                    logging.info(f"Loaded master data from pickle: {self.current_data.shape}")
                    return True
            
            # 2. pickle이 없으면 wage_increase.xlsx에서 로드
            if os.path.exists(self.master_excel_file):
                logging.info(f"Loading master data from {self.master_excel_file}")
                self.load_data_from_file(self.master_excel_file, save_file=False)
                return True
                
        except Exception as e:
            logging.warning(f"Failed to load master data: {e}")
        return False
    
    def _save_data_to_pickle(self) -> None:
        """현재 데이터를 pickle 파일로 저장"""
        try:
            if self.current_data is not None and self.data_info is not None:
                data_package = {
                    'data': self.current_data,
                    'info': self.data_info,
                    'timestamp': datetime.now().isoformat()
                }
                with open(self.pickle_file, 'wb') as f:
                    pickle.dump(data_package, f)
                logging.info(f"Saved data to pickle: {self.current_data.shape}")
        except Exception as e:
            logging.error(f"Failed to save data to pickle: {e}")
    
    def get_default_data_status(self) -> Dict[str, Any]:
        """마스터 데이터 상태 확인"""
        return {
            "has_master_data": self.current_data is not None,
            "pickle_exists": self.pickle_file.exists(),
            "master_excel_exists": os.path.exists(self.master_excel_file),
            "data_shape": self.current_data.shape if self.current_data is not None else None,
            "pickle_file_path": str(self.pickle_file),
            "master_excel_path": self.master_excel_file
        }
    
    def save_uploaded_file(self, file_content: bytes, filename: str) -> str:
        """업로드된 파일을 임시로 처리 (저장하지 않음)"""
        # 파일을 저장하지 않고 임시 파일 경로만 반환
        # 실제로는 메모리에서 직접 처리
        import tempfile
        
        # 임시 파일 생성
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as tmp_file:
            tmp_file.write(file_content)
            temp_path = tmp_file.name
        
        return temp_path
    
    def load_data_from_file(self, file_path: str, save_file: bool = True) -> Dict[str, Any]:
        """파일에서 데이터 로드 및 분석"""
        try:
            # 파일 확장자에 따라 적절한 로더 사용
            if file_path.endswith(('.xlsx', '.xls')):
                # Excel 파일 읽기
                # SambaWage_250825.xlsx의 경우: 첫 번째 행은 한글 헤더, 두 번째 행은 영문 헤더
                if 'SambaWage' in file_path:
                    df = pd.read_excel(file_path, header=1)  # 두 번째 행(영문)을 헤더로 사용
                else:
                    df = pd.read_excel(file_path, header=0)  # 일반 파일은 첫 번째 행을 헤더로
                
                # 만약 첫 번째 데이터 행이 영문 헤더가 아닌 실제 데이터인지 확인
                # (year 컬럼이 있고 첫 번째 값이 숫자여야 함)
                if 'year' in df.columns:
                    # year 컬럼의 첫 번째 값이 문자열인지 확인
                    first_year = df.iloc[0]['year'] if len(df) > 0 else None
                    if first_year is not None:
                        try:
                            # 숫자로 변환 시도
                            float(first_year)
                        except (ValueError, TypeError):
                            # 첫 번째 행이 데이터가 아닌 경우 (예: 또 다른 헤더)
                            # 이 행을 제거
                            df = df.iloc[1:].reset_index(drop=True)
                            logging.info("Removed additional header row from Excel data")
                
                # 데이터 타입 변환 (숫자 컬럼을 float로)
                for col in df.columns:
                    if col != 'year':  # year 제외한 모든 컬럼
                        try:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                        except:
                            pass
                
                logging.info(f"Loaded Excel file with shape: {df.shape}, columns: {list(df.columns)[:5]}...")
                
            elif file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
            
            # 데이터 저장
            self.current_data = df
            
            # 데이터 정보 생성
            data_info = self._analyze_dataframe(df)
            self.data_info = data_info
            
            # pickle 파일로 저장 (save_file이 True일 때만)
            if save_file:
                self._save_data_to_pickle()
            
            # 임시 파일인 경우 삭제
            if save_file and file_path.startswith('/tmp') or file_path.startswith('/var'):
                try:
                    os.remove(file_path)
                except:
                    pass
            
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
        
        # 샘플 데이터
        sample_data = df.head(10).fillna("").to_dict(orient="records")
        
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
    
    def augment_data_with_noise(self, target_size: int = 120, noise_factor: float = 0.01) -> Dict[str, Any]:
        """Target 컬럼을 제외하고 나머지 데이터에 대해서 10배수로 증강"""
        if self.current_data is None:
            raise ValueError("No data loaded for augmentation")
        
        # 재현성을 위한 랜덤 시드 고정 - 이 함수에서만 로컬하게 사용
        rng = np.random.RandomState(42)  # 전역 상태를 변경하지 않음
        
        original_df = self.current_data.copy()
        original_size = len(original_df)
        
        # Target 컬럼 식별 (대소문자 무시하고 target이 포함된 컬럼 찾기)
        target_columns = [col for col in original_df.columns if 'target' in col.lower()]
        
        print(f"📊 Data Augmentation (Target 제외 10배수):")
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
                            noise = rng.normal(0, abs(new_row[col]) * noise_factor)
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