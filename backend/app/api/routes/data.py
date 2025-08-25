from fastapi import APIRouter, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
from typing import Dict, Any, Optional
from pydantic import BaseModel
import os
from app.core.config import settings
from app.services.data_service import data_service

router = APIRouter()

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
        print(f"📁 Processing file: {file.filename}")
        
        # 파일 내용 읽기
        contents = await file.read()
        print(f"📊 File size: {len(contents)} bytes")
        
        # 파일 저장
        saved_path = data_service.save_uploaded_file(contents, file.filename)
        print(f"💾 File saved to: {saved_path}")
        
        # 데이터 로드 및 분석
        data_info = data_service.load_data_from_file(saved_path)
        print(f"📈 Data loaded: {data_info['basic_stats']['shape']}")
        
        # 모델링 준비 상태 확인
        validation_result = data_service.validate_data_for_modeling()
        print(f"✅ Validation: {validation_result.get('is_valid', False)}")
        
        # 요약 정보 생성
        summary = data_service.get_data_summary()
        print(f"📋 Summary generated: {summary['shape']}")
        
        return {
            "message": "File uploaded and processed successfully",
            "filename": file.filename,
            "data_analysis": data_info,
            "validation": validation_result,
            "summary": summary
        }
        
    except Exception as e:
        print(f"❌ Error processing file: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@router.get("/master")
async def get_master_data(rows: int = Query(default=10, ge=1, le=1000)) -> Dict[str, Any]:
    """
    마스터 데이터 반환 (wage_increase.xlsx 파일 기반)
    """
    try:
        # 기존 wage_increase.xlsx 파일이 있는지 확인
        sample_file = "wage_increase.xlsx"
        if os.path.exists(sample_file):
            # 샘플 파일 로드
            data_info = data_service.load_data_from_file(sample_file)
            sample_data = data_service.get_sample_data(rows)
            
            return {
                "message": "Master data loaded successfully",
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
            "current_data_shape": status["data_shape"],
            "has_master_data": status.get("has_master_data", status.get("has_default_data", False)),
            "master_data_source": "pickle" if status.get("pickle_exists") else "excel" if status.get("master_excel_exists") else "none"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting data status: {str(e)}")

@router.post("/load-master")
async def load_master_data() -> Dict[str, Any]:
    """
    마스터 데이터 로드 (pickle 파일 또는 wage_increase.xlsx에서)
    """
    try:
        success = data_service._load_master_data()
        if success:
            summary = data_service.get_data_summary()
            return {
                "message": "Master data loaded successfully",
                "summary": summary,
                "source": "pickle" if data_service.pickle_file.exists() else "wage_increase.xlsx"
            }
        else:
            raise HTTPException(status_code=404, detail="No master data available. Please ensure wage_increase.xlsx exists.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading master data: {str(e)}")

@router.post("/augment")
async def augment_data(request: DataAugmentationRequest) -> Dict[str, Any]:
    """
    데이터 증강 - 노이즈 기반으로 120개 데이터로 확장
    """
    try:
        if data_service.current_data is None:
            raise HTTPException(status_code=404, detail="No data loaded for augmentation")
        
        result = data_service.augment_data_with_noise(
            target_size=request.target_size,
            noise_factor=request.noise_factor
        )
        
        # 증강 후 요약 정보 추가
        if result["augmentation_applied"]:
            result["summary"] = data_service.get_data_summary()
        
        return result
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error augmenting data: {str(e)}")