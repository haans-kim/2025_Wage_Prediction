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
        self.pickle_file = self.data_dir / "current_data.pkl"
        self.current_data: Optional[pd.DataFrame] = None
        self.data_info: Optional[Dict[str, Any]] = None
        
        # 시작시 기본 데이터 로드 시도
        self._load_default_data()
    
    def _load_default_data(self) -> bool:
        """기본 데이터 로드 (pickle 파일에서)"""
        try:
            if self.pickle_file.exists():
                with open(self.pickle_file, 'rb') as f:
                    data_package = pickle.load(f)
                    self.current_data = data_package['data']
                    self.data_info = data_package['info']
                    logging.info(f"Loaded default data from pickle: {self.current_data.shape}")
                    return True
        except Exception as e:
            logging.warning(f"Failed to load default data from pickle: {e}")
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
        """기본 데이터 상태 확인"""
        return {
            "has_default_data": self.current_data is not None,
            "pickle_exists": self.pickle_file.exists(),
            "data_shape": self.current_data.shape if self.current_data is not None else None,
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
            
            # 데이터 저장
            self.current_data = df
            
            # 데이터 정보 생성
            data_info = self._analyze_dataframe(df)
            self.data_info = data_info
            
            # pickle 파일로 저장
            self._save_data_to_pickle()
            
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