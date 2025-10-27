# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for SambioWage Backend Server
FastAPI + PyCaret + ML libraries를 단일 실행 파일로 번들링
"""

import sys
from PyInstaller.utils.hooks import collect_all, collect_submodules

# 수집할 패키지들
packages_to_collect = [
    'pycaret',
    'sklearn',
    'sktime',
    'shap',
    'lime',
    'fastapi',
    'uvicorn',
    'starlette',
    'pydantic',
    'pydantic_core',
    'explainerdashboard',
    'plotly',
    'dash',
    'pandas',
    'numpy',
    'scipy',
    'statsmodels',
    'lightgbm',
    'xgboost',
]

datas = []
binaries = []
hiddenimports = []

# 각 패키지의 모든 데이터, 바이너리, 숨겨진 imports 수집
for package in packages_to_collect:
    try:
        datas_collected, binaries_collected, hiddenimports_collected = collect_all(package)
        datas += datas_collected
        binaries += binaries_collected
        hiddenimports += hiddenimports_collected
    except Exception as e:
        print(f"Warning: Could not collect {package}: {e}")

# 추가 숨겨진 imports (PyCaret 및 scikit-learn 관련)
additional_hiddenimports = [
    # PyCaret
    'pycaret.regression',
    'pycaret.classification',
    'pycaret.clustering',

    # Scikit-learn
    'sklearn.utils._typedefs',
    'sklearn.utils._heap',
    'sklearn.utils._sorting',
    'sklearn.utils._vector_sentinel',
    'sklearn.neighbors._partition_nodes',
    'sklearn.tree._utils',

    # SHAP
    'shap.explainers',
    'shap.explainers._tree',
    'shap.explainers._linear',
    'shap.explainers._deep',
    'shap.plots',

    # LIME
    'lime.lime_tabular',
    'lime.lime_text',
    'lime.lime_image',

    # FastAPI / Uvicorn
    'uvicorn.logging',
    'uvicorn.loops',
    'uvicorn.loops.auto',
    'uvicorn.protocols',
    'uvicorn.protocols.http',
    'uvicorn.protocols.http.auto',
    'uvicorn.protocols.websockets',
    'uvicorn.protocols.websockets.auto',
    'uvicorn.lifespan',
    'uvicorn.lifespan.on',

    # Pydantic
    'pydantic.types',
    'pydantic.fields',
    'pydantic_settings',

    # Explainer Dashboard
    'explainerdashboard.explainers',
    'explainerdashboard.dashboards',

    # Plotly
    'plotly.graph_objs',
    'plotly.graph_objects',

    # Stats
    'statsmodels.tsa',
    'statsmodels.tsa.api',

    # Numba (PyCaret 의존성)
    'numba',
    'numba.core',
    'numba.typed',
]

hiddenimports += additional_hiddenimports

# 제외할 모듈 (불필요한 크기 절감)
excludes = [
    'matplotlib.tests',
    'numpy.tests',
    'pandas.tests',
    'scipy.tests',
    'sklearn.tests',
    'IPython',
    'notebook',
    'jupyter',
    'pytest',
    'sphinx',
    'setuptools',
    'distutils',
    'tkinter',
    'PySide2',
    'PyQt5',
    'wx',
]

a = Analysis(
    ['run.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='backend-server',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,  # 콘솔 창 표시 (디버깅용, 나중에 False로 변경 가능)
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # 아이콘 파일 경로 (선택사항)
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='backend-server',
)
