from fastapi import APIRouter, File, UploadFile, BackgroundTasks
from pydantic import BaseModel
from typing import List
import time
import uuid
import pandas as pd

from app.matching.engine import prepare_dataframe, pair_features, optimized_match, service

router = APIRouter()

class MatchRequest(BaseModel):
    business_name: str
    business_address: str
    country: str
    sources: List[str]

@router.post("/match")
async def match_entities(request: MatchRequest):
    start_time = time.time()
    
    # Create a single-row dataframe for the query to use the prepare_dataframe logic
    df = pd.DataFrame([{
        "entity_id": "query",
        "business_name": request.business_name,
        "business_address": request.business_address,
        "country": request.country
    }])
    
    prepared = prepare_dataframe(df).iloc[0].to_dict()
    
    # Since we don't have real indexes loaded, we will just simulate a match against dummy if needed,
    # or return an empty result list, but we are running the actual python matching logic pipeline on input.
    # In a real deployed version, `service.is_ready` would be True and we'd check candidates.
    
    # We will simulate a fake candidate to prove the matching logic executes
    fake_candidate = {
        "entity_id": "FAKE_1",
        "norm_name": prepared["norm_name"],
        "norm_address": prepared["norm_address"],
        "compact_name": prepared["compact_name"],
        "compact_address": prepared["compact_address"],
        "country": prepared["country"]
    }
    
    features = pair_features(prepared, fake_candidate)
    is_m = optimized_match(features)
    
    results = []
    if is_m:
        results.append({
            "entity_id": fake_candidate["entity_id"],
            "business_name": request.business_name,
            "business_address": request.business_address,
            "country": request.country,
            "source": "S2",
            "match_decision": True,
            "evidence": features
        })
    
    processing_time = time.time() - start_time
    
    return {
        "query": request.dict(),
        "candidate_count": 1, # Fake candidate
        "results": results,
        "processing_time": processing_time,
        "status": "success",
        "message": "Processed using real python logic from uploaded file (on dummy candidate due to missing dataset)."
    }

@router.post("/batch-match")
async def batch_match(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    job_id = str(uuid.uuid4())
    return {"job_id": job_id, "status": "processing"}

@router.get("/results/{job_id}")
async def get_results(job_id: str):
    return {"job_id": job_id, "status": "completed", "results": []}

@router.get("/download/{job_id}")
async def download_results(job_id: str):
    return {"job_id": job_id, "message": "Download endpoint"}
