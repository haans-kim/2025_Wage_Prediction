"""
Cleanup utility for removing old pickle files and keeping only the latest ones
"""
import os
import glob
import logging
from pathlib import Path


def cleanup_old_pickle_files():
    """Remove all old pickle files, keeping only the latest ones"""

    # Backend 디렉토리 경로
    backend_dir = Path(__file__).parent.parent.parent

    # 1. models 디렉토리의 오래된 파일들 삭제
    models_dir = backend_dir / 'models'
    if models_dir.exists():
        model_files = glob.glob(str(models_dir / '*.pkl'))

        # latest.pkl 제외한 모든 파일 삭제
        for file_path in model_files:
            if 'latest' not in os.path.basename(file_path):
                try:
                    os.remove(file_path)
                    logging.info(f"Removed old model file: {os.path.basename(file_path)}")
                except Exception as e:
                    logging.error(f"Failed to remove {file_path}: {e}")

    # 2. data 디렉토리의 오래된 파일들 확인 (master_data.pkl, working_data.pkl 유지)
    data_dir = backend_dir / 'data'
    if data_dir.exists():
        data_files = glob.glob(str(data_dir / '*.pkl'))

        # master_data.pkl과 working_data.pkl 이외의 파일들 삭제
        keep_files = ['master_data.pkl', 'working_data.pkl']
        for file_path in data_files:
            file_name = os.path.basename(file_path)
            if file_name not in keep_files:
                try:
                    os.remove(file_path)
                    logging.info(f"Removed old data file: {file_name}")
                except Exception as e:
                    logging.error(f"Failed to remove {file_path}: {e}")

    print("[OK] Cleanup completed - keeping only latest pickle files")


def get_pickle_files_status():
    """Get status of all pickle files in the backend"""

    backend_dir = Path(__file__).parent.parent.parent
    status = {
        'models': [],
        'data': []
    }

    # Check models directory
    models_dir = backend_dir / 'models'
    if models_dir.exists():
        model_files = glob.glob(str(models_dir / '*.pkl'))
        for file_path in model_files:
            file_stats = os.stat(file_path)
            status['models'].append({
                'name': os.path.basename(file_path),
                'size_mb': round(file_stats.st_size / (1024 * 1024), 2),
                'path': file_path
            })

    # Check data directory
    data_dir = backend_dir / 'data'
    if data_dir.exists():
        data_files = glob.glob(str(data_dir / '*.pkl'))
        for file_path in data_files:
            file_stats = os.stat(file_path)
            status['data'].append({
                'name': os.path.basename(file_path),
                'size_mb': round(file_stats.st_size / (1024 * 1024), 2),
                'path': file_path
            })

    return status


if __name__ == "__main__":
    # 수동 실행 시 cleanup 수행
    print("Starting cleanup of old pickle files...")
    cleanup_old_pickle_files()

    print("\nCurrent pickle files status:")
    status = get_pickle_files_status()
    print(f"Models: {len(status['models'])} files")
    for model in status['models']:
        print(f"  - {model['name']} ({model['size_mb']} MB)")

    print(f"\nData: {len(status['data'])} files")
    for data in status['data']:
        print(f"  - {data['name']} ({data['size_mb']} MB)")