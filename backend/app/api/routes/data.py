from fastapi import APIRouter, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
from typing import Dict, Any, Optional
from pydantic import BaseModel
import os
from app.core.config import settings
from app.services.data_service import data_service
from app.services.augmentation_service import augmentation_service
from app.utils.cleanup import cleanup_old_pickle_files, get_pickle_files_status

router = APIRouter()

class AugmentationRequest(BaseModel):
    method: str = 'auto'  # 'auto', 'noise', 'interpolation', 'mixup'
    factor: Optional[int] = None  # 증강 배수
    target_size: Optional[int] = None  # 목표 크기
    noise_level: float = 0.02  # 노이즈 수준
    preserve_distribution: bool = True  # 분포 유지

@router.options("/upload")
async def upload_options():
    return {"message": "OK"}

@router.post("/upload")
async def upload_data(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Excel/CSV 파일 업로드 및 데이터 분석
    """
    # 파일 크기 검증
    if file.size and file.size > settings.MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413, 
            detail=f"File size exceeds maximum allowed size ({settings.MAX_FILE_SIZE} bytes)"
        )
    
    # 파일명 검증
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    # 파일 확장자 검증
    if not any(file.filename.endswith(ext) for ext in settings.ALLOWED_FILE_TYPES):
        raise HTTPException(
            status_code=400,
            detail=f"File type not allowed. Allowed types: {settings.ALLOWED_FILE_TYPES}"
        )
    
    try:
        print(f"[DATA] Processing file: {file.filename}")

        # 파일 내용 읽기
        contents = await file.read()
        print(f"[DATA] File size: {len(contents)} bytes")

        # 파일 저장
        saved_path = data_service.save_uploaded_file(contents, file.filename)
        print(f"[DATA] File saved to: {saved_path}")

        # 데이터 로드 및 분석
        data_info = data_service.load_data_from_file(saved_path)
        print(f"[DATA] Data loaded: {data_info['basic_stats']['shape']}")

        # 데이터 변경으로 인한 모델 자동 초기화
        data_service._clear_models_on_data_change()

        # 모델링 준비 상태 확인
        validation_result = data_service.validate_data_for_modeling()
        print(f"[DATA] Validation: {validation_result.get('is_valid', False)}")

        # 요약 정보 생성
        summary = data_service.get_data_summary()
        print(f"[DATA] Summary generated: {summary['shape']}")
        
        # 모델 설정 정보 추가
        model_config = data_service.get_model_config()
        
        return {
            "message": "File uploaded and processed successfully",
            "filename": file.filename,
            "file_path": saved_path,
            "data_analysis": data_info,
            "validation": validation_result,
            "summary": summary,
            "model_config": model_config
        }
        
    except Exception as e:
        print(f"[ERROR] Error processing file: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@router.get("/sample")
async def get_sample_data(rows: int = Query(default=10, ge=1, le=1000)) -> Dict[str, Any]:
    """
    샘플 데이터 반환 (wage_increase.xlsx 파일 기반)
    """
    try:
        # 기존 wage_increase.xlsx 파일이 있는지 확인
        sample_file = "wage_increase.xlsx"
        if os.path.exists(sample_file):
            # 샘플 파일 로드
            data_info = data_service.load_data_from_file(sample_file)
            sample_data = data_service.get_sample_data(rows)
            
            return {
                "message": "Sample data loaded successfully",
                "source": "wage_increase.xlsx",
                "data": sample_data["data"],
                "summary": data_service.get_data_summary(),
                "analysis": {
                    "basic_stats": data_info["basic_stats"],
                    "missing_analysis": data_info["missing_analysis"]
                }
            }
        else:
            # 샘플 파일이 없는 경우 가상 데이터 생성
            import pandas as pd
            import numpy as np
            
            # 가상의 임금 데이터 생성
            np.random.seed(42)
            sample_df = pd.DataFrame({
                'year': range(2014, 2024),
                'inflation_rate': np.random.normal(2.5, 1.0, 10),
                'gdp_growth': np.random.normal(2.8, 1.5, 10),
                'unemployment_rate': np.random.normal(3.5, 0.8, 10),
                'wage_increase_rate': np.random.normal(3.2, 1.2, 10),
                'productivity_index': np.random.normal(100, 10, 10)
            })
            
            return {
                "message": "Generated sample data (demo)",
                "source": "generated",
                "data": sample_df.round(2).to_dict(orient="records"),
                "shape": sample_df.shape,
                "columns": sample_df.columns.tolist()
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading sample data: {str(e)}")

@router.get("/current")
async def get_current_data(
    rows: int = Query(default=20, ge=1, le=1000),
    include_analysis: bool = Query(default=False)
) -> Dict[str, Any]:
    """
    현재 로드된 데이터 반환
    """
    try:
        if data_service.current_data is None:
            raise HTTPException(status_code=404, detail="No data currently loaded")
        
        sample_data = data_service.get_sample_data(rows)
        summary = data_service.get_data_summary()
        
        result = {
            "message": "Current data retrieved successfully",
            "data": sample_data["data"],
            "summary": summary
        }
        
        if include_analysis and data_service.data_info:
            result["analysis"] = data_service.data_info
        
        return result
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving current data: {str(e)}")

@router.get("/validate")
async def validate_current_data() -> Dict[str, Any]:
    """
    현재 데이터의 모델링 준비 상태 검증
    """
    try:
        if data_service.current_data is None:
            raise HTTPException(status_code=404, detail="No data currently loaded")
        
        validation_result = data_service.validate_data_for_modeling()
        
        return {
            "message": "Data validation completed",
            "validation": validation_result,
            "summary": data_service.get_data_summary()
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error validating data: {str(e)}")

@router.get("/columns")
async def get_data_columns():
    """현재 로드된 데이터의 컬럼 정보 반환"""
    if data_service.current_data is None:
        raise HTTPException(status_code=404, detail="No data loaded")

    columns = list(data_service.current_data.columns)
    numeric_columns = list(data_service.current_data.select_dtypes(include=['int64', 'float64']).columns)

    # Feature importance를 위한 컬럼 분석 (타겟 컬럼 제외)
    target_column = None
    for col in reversed(columns):
        if 'increase' in col.lower() or 'rate' in col.lower() or '인상' in col:
            target_column = col
            break

    feature_columns = [col for col in numeric_columns if col != target_column]

    return {
        "all_columns": columns,
        "numeric_columns": numeric_columns,
        "feature_columns": feature_columns,
        "target_column": target_column,
        "shape": data_service.current_data.shape
    }


@router.get("/info")
async def get_data_info() -> Dict[str, Any]:
    """
    데이터 업로드 시스템 정보 반환
    """
    return {
        "supported_formats": settings.ALLOWED_FILE_TYPES,
        "max_file_size_mb": settings.MAX_FILE_SIZE / 1024 / 1024,
        "upload_directory": settings.UPLOAD_DIR,
        "current_data_loaded": data_service.current_data is not None,
        "system_status": "ready"
    }

@router.delete("/current")
async def clear_current_data() -> Dict[str, Any]:
    """
    현재 로드된 데이터 삭제
    """
    data_service.current_data = None
    data_service.data_info = None
    
    return {
        "message": "Current data cleared successfully"
    }

class DataAugmentationRequest(BaseModel):
    target_size: int = 120
    noise_factor: float = 0.02

@router.get("/status")
async def get_data_status() -> Dict[str, Any]:
    """
    데이터 상태 및 기본 데이터 로드 정보
    """
    try:
        status = data_service.get_default_data_status()
        return {
            "message": "Data status retrieved successfully",
            "status": status,
            "master_data_shape": status["master_data_shape"],
            "working_data_shape": status["working_data_shape"],
            "has_master_data": status["has_master_data"],
            "is_augmented": status["is_augmented"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting data status: {str(e)}")

@router.post("/cleanup")
async def cleanup_pickle_files() -> Dict[str, Any]:
    """
    오래된 pickle 파일들을 정리하고 최신 파일만 유지
    """
    try:
        # 정리 전 상태
        before_status = get_pickle_files_status()

        # 정리 실행
        cleanup_old_pickle_files()

        # 정리 후 상태
        after_status = get_pickle_files_status()

        return {
            "message": "Cleanup completed successfully",
            "before": {
                "model_files": len(before_status["models"]),
                "data_files": len(before_status["data"])
            },
            "after": {
                "model_files": len(after_status["models"]),
                "data_files": len(after_status["data"])
            },
            "removed": {
                "model_files": len(before_status["models"]) - len(after_status["models"]),
                "data_files": len(before_status["data"]) - len(after_status["data"])
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during cleanup: {str(e)}")

@router.get("/pickle-status")
async def get_pickle_status() -> Dict[str, Any]:
    """
    현재 pickle 파일들의 상태 조회
    """
    try:
        status = get_pickle_files_status()
        return {
            "message": "Pickle files status retrieved successfully",
            "models": status["models"],
            "data": status["data"],
            "total_files": len(status["models"]) + len(status["data"])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting pickle status: {str(e)}")

@router.post("/load-default")
async def load_default_data() -> Dict[str, Any]:
    """
    기본 데이터 로드 (pickle 파일에서)
    """
    try:
        success = data_service._load_default_data()
        if success:
            # 데이터 변경으로 인한 모델 자동 초기화
            data_service._clear_models_on_data_change()

            summary = data_service.get_data_summary()
            return {
                "message": "Default data loaded successfully",
                "summary": summary,
                "loaded_from_pickle": True
            }
        else:
            raise HTTPException(status_code=404, detail="No default data available")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading default data: {str(e)}")

@router.post("/augment")
async def augment_data(request: AugmentationRequest) -> Dict[str, Any]:
    """
    데이터 증강 - 마스터 데이터를 보존하고 작업 데이터만 증강
    """
    try:
        if data_service.master_data is None:
            raise HTTPException(status_code=404, detail="No master data loaded for augmentation")
        
        # 모델 설정 가져오기
        model_config = data_service.get_model_config()
        target_column = model_config.get('target_column')
        year_column = model_config.get('year_column')
        
        # 마스터 데이터로부터 증강 (복사본 사용)
        augmented_df, info = augmentation_service.smart_augment(
            df=data_service.master_data.copy(),  # 마스터 데이터의 복사본 사용
            target_column=target_column,
            year_column=year_column,
            target_size=request.target_size,
            method=request.method
        )
        
        # 작업 데이터만 업데이트 (마스터는 보존)
        data_service.current_data = augmented_df
        data_service.last_augmentation_info = info
        data_service._save_working_data_to_pickle()  # 작업 데이터만 저장
        
        # 결과 반환
        return {
            "message": f"Data augmented successfully using {info['method']}",
            "augmentation_info": info,
            "master_preserved": True,
            "master_shape": data_service.master_data.shape,
            "working_shape": data_service.current_data.shape,
            "summary": data_service.get_data_summary()
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error augmenting data: {str(e)}")

@router.post("/reset-to-master")
async def reset_to_master() -> Dict[str, Any]:
    """
    작업 데이터를 마스터 데이터로 리셋
    """
    try:
        result = data_service.reset_to_master()
        return {
            "message": "Successfully reset to master data",
            **result,
            "summary": data_service.get_data_summary()
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error resetting data: {str(e)}")