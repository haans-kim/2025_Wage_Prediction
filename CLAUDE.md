# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SambioWage is a wage prediction application using machine learning (PyCaret) with a React frontend and FastAPI backend. The application performs wage increase rate predictions using various regression models.

## Development Commands

### Virtual Environment Setup (Python 3.10)
```bash
# Create virtual environment with Python 3.10
python3.10 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install backend dependencies
pip install -r backend/requirements.txt
```

### Frontend (React + TypeScript)
```bash
cd frontend
npm install              # Install dependencies
npm start                # Start development server (port 3001)
npm run build           # Build for production
npm test                # Run tests
```

### Backend (FastAPI + PyCaret)
```bash
cd backend
source ../venv/bin/activate        # Activate virtual environment
python run.py                      # Start server (port 8001)
# OR
uvicorn app.main:app --reload --port 8001
```

### Full Stack Development
```bash
# Using Docker Compose
docker-compose up -d

# Deploy script for various platforms
./deploy.sh
```

## Architecture

### Frontend Structure
- **React Router**: Navigation between pages (DataUpload, Modeling, Analysis, Dashboard, Effects, ExplainerDashboard)
- **Component Organization**: 
  - `/components/layout/`: Layout and Sidebar components
  - `/components/ui/`: Reusable UI components (alert, button, card, tabs)
  - `/pages/`: Main application pages
  - `/lib/api.ts`: API client for backend communication

### Backend Structure
- **FastAPI Application**: Main entry at `backend/app/main.py`
- **Service Layer**: Business logic in `/services/`
  - `data_service.py`: Data loading and preprocessing
  - `modeling_service.py`: PyCaret model training and evaluation
  - `analysis_service.py`: Model analysis and predictions
  - `dashboard_service.py`: Dashboard data generation
  - `explainer_dashboard_service.py`: Model explainability
- **API Routes**: RESTful endpoints in `/api/routes/`
- **Data Storage**: Pickle files for master data, Excel files for raw data

### Key Technical Details

1. **Model Training Flow**:
   - Data upload → Preprocessing → PyCaret setup → Model comparison → Training → Evaluation
   - Automatic model selection based on data size (small/medium/large)
   - Support for regression models: lr, ridge, lasso, en, dt, rf, gbr, xgboost, lightgbm

2. **Data Management**:
   - Master data stored in `data/master_data.pkl`
   - Uploaded files processed in memory or saved to `uploads/`
   - Excel files (`.xlsx`) as primary data format

3. **Port Configuration**:
   - Frontend: Port 3001 (changed from default 3000)
   - Backend: Port 8001 (changed from default 8000)
   - Configuration in `frontend/.env` and `backend/run.py`

4. **CORS Configuration**:
   - Backend configured to accept all origins (`*`)
   - Double CORS middleware for compatibility

5. **Deployment Options**:
   - Railway (recommended - easiest)
   - Render
   - Docker Compose
   - Fly.io
   - Vercel (frontend) + separate backend

## API Endpoints

- `GET /`: API root
- `GET /health`: Health check
- `/api/data/*`: Data management endpoints
- `/api/modeling/*`: Model training and management
- `/api/analysis/*`: Analysis and predictions
- `/api/dashboard/*`: Dashboard data generation

## Important Files

- `backend/app/main.py`: FastAPI application entry point
- `backend/app/services/modeling_service.py`: Core ML logic using PyCaret
- `backend/app/services/data_service.py`: Data management logic
- `frontend/src/App.tsx`: React app routing
- `frontend/src/lib/api.ts`: Backend API client
- `docker-compose.yml`: Full stack local deployment
- `deploy.sh`: Deployment helper script