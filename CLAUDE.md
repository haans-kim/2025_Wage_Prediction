# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
SambioWage - A machine learning-based wage increase prediction dashboard with React frontend and FastAPI backend.

## Development Commands

### Quick Start
```bash
# Install dependencies and start both frontend and backend
make install
make start        # Smart port detection
make start-fixed  # Fixed ports (3000 frontend, 8000 backend)
make restart      # Clean restart
```

### Backend Commands
```bash
cd backend
source venv/bin/activate  # Activate virtual environment
pip install -r requirements.txt
python run.py  # Runs on localhost:8000
```

### Frontend Commands
```bash
cd frontend
npm install
npm start     # Runs on localhost:3000
npm run build # Production build
npm test      # Run tests
```

### Other Commands
```bash
make stop         # Stop all services
make logs         # View backend logs
make check-ports  # Check port status
make clean        # Reset development environment
```

### Docker Commands
```bash
docker-compose up -d    # Start with Docker
docker-compose down     # Stop containers
docker-compose logs -f  # View logs
```

## Architecture

### Frontend (React + TypeScript)
- **Entry Point**: `frontend/src/index.tsx`
- **Main App**: `frontend/src/App.tsx` - React Router setup with main routes
- **Pages**: Located in `frontend/src/pages/`
  - DataUpload.tsx - Data upload and validation
  - Modeling.tsx - ML model training interface
  - Analysis.tsx - Model analysis and explainability
  - Dashboard.tsx - Main prediction dashboard
  - ExplainerDashboard.tsx - Detailed model explanations
  - Effects.tsx - Variable effects visualization
- **API Client**: `frontend/src/lib/api.ts` - Centralized API communication layer
- **Components**: `frontend/src/components/` - Reusable UI components
- **Styling**: Tailwind CSS with custom components

### Backend (FastAPI + PyCaret)
- **Entry Point**: `backend/run.py` - Uvicorn server initialization
- **Main App**: `backend/app/main.py` - FastAPI app configuration and routing
- **API Routes**: `backend/app/api/routes/`
  - data.py - Data upload, validation, and management
  - modeling.py - ML model training and evaluation
  - analysis.py - Model explainability (SHAP, LIME)
  - dashboard.py - Prediction and scenario analysis
- **Services**: `backend/app/services/` - Business logic layer
  - data_service.py - Data processing and storage
  - modeling_service.py - PyCaret integration for ML
  - analysis_service.py - Model interpretation
  - dashboard_service.py - Prediction logic
  - explainer_dashboard_service.py - ExplainerDashboard generation
- **Data Storage**: Files stored in `backend/uploads/` and `backend/data/`

## Key Technical Details

### Frontend-Backend Communication
- Base URL configured via `REACT_APP_API_URL` environment variable
- Default: `http://localhost:8000`
- CORS enabled for all origins in development

### Machine Learning Pipeline
- **Framework**: PyCaret for AutoML
- **Models**: Linear Regression, Ridge, Lasso, Random Forest, XGBoost, LightGBM
- **Data Requirements**: Minimum 20 rows, adapts model selection based on data size
- **Target Column**: Configurable, typically wage increase percentage

### State Management
- Frontend uses React hooks (useState, useEffect)
- Backend maintains session state for ML experiments
- File uploads persisted in `backend/uploads/`

### API Endpoints Structure
- `/api/data/*` - Data management
- `/api/modeling/*` - Model training and evaluation
- `/api/analysis/*` - Model explainability
- `/api/dashboard/*` - Predictions and scenarios

## Important Files and Their Purposes

- `Makefile` - Development automation commands
- `docker-compose.yml` - Container orchestration
- `backend/requirements.txt` - Python dependencies
- `frontend/package.json` - Node dependencies
- `.cursor/rules/*.mdc` - Cursor AI development rules

## Deployment

### Frontend (Vercel)
- Auto-deploys from Git repository
- Root directory: `/frontend`
- Build command: `npm run build`
- Output directory: `build`

### Backend Options
- Railway (recommended) - Supports Python and persistent storage
- Render - Good for FastAPI apps
- Docker deployment supported via included Dockerfiles

## Testing Approach
- Frontend: React Testing Library (`npm test`)
- Backend: Manual testing via API endpoints
- No automated backend tests currently implemented

## Claude Code Guidelines

### Important Rules
- **NEVER use emojis in any code, logs, or comments**
- All logging and debugging output must use plain ASCII text only
- Replace emojis with descriptive text like [OK], [WARN], [ERROR], [INFO], [DATA], etc.
- Windows cp949 encoding is incompatible with emoji characters - always use ASCII alternatives