# Amazon ML Challenge 2026 - Entity Resolution

## Project Structure
- `backend/`: FastAPI application to serve the entity resolution model.
- `frontend/`: React + Vite UI.
- `notebooks/`: Put `amazon-ml-challenge-2026-entity-resolution.ipynb` here.

## Setup Instructions

### Backend
1. Place your original notebook in `notebooks/amazon-ml-challenge-2026-entity-resolution.ipynb`.
2. Extract the ML logic (`prepare_small()`, `match_score()`, etc.) into `backend/app/api/match.py` or a dedicated ML service file.
3. Install dependencies:
   ```bash
   cd backend
   pip install -r requirements.txt
   ```
4. Run server:
   ```bash
   uvicorn app.main:app --reload
   ```

### Frontend
1. Install dependencies:
   ```bash
   cd frontend
   npm install
   ```
2. Start dev server:
   ```bash
   npm run dev
   ```
3. Set `.env` variable `VITE_API_BASE_URL` to the backend URL (e.g., ngrok URL if using Colab).

## Note on the Notebook
The provided `amazon-ml-challenge-2026-entity-resolution.ipynb` was not found in the workspace. Please move it to the `notebooks` directory. The backend API is currently using placeholder logic and is ready to be connected to the actual python matching pipeline once it's available.
