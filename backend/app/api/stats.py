from fastapi import APIRouter

router = APIRouter()

@router.get("/statistics")
async def get_statistics():
    return {
        "total_source1_entities": 0,
        "total_source2_entities": 0,
        "total_source3_entities": 0,
        "candidate_pairs_generated": 0,
        "matched_entities": 0,
        "unmatched_entities": 0,
        "average_processing_time": 0.0
    }
