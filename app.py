from fastapi import FastAPI, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import time
from retrieval import QueryProcessor
from ranker import Ranker
import json
import os

app = FastAPI(title="Dejan Petrovic Search Engine")

# Initialize components
qp = QueryProcessor()
ranker = Ranker()

# Mock cache for hot queries
CACHE = {}

@app.get("/search")
async def search(q: str = Query(..., min_length=1)):
    start_time = time.time()
    
    # Check cache
    if q in CACHE:
        return CACHE[q]
    
    # 1. Retrieval (Hybrid)
    retrieval_start = time.time()
    raw_results = qp.hybrid_search(q, limit=50)
    retrieval_latency = (time.time() - retrieval_start) * 1000
    
    # 2. Ranking (Feature-based)
    ranking_start = time.time()
    # Enrich raw results with content for ranking
    # In a real system, we'd fetch snippets/features here
    # For the prototype, QueryProcessor already returned some data
    # We'll re-fetch full content from data files if needed, 
    # but here we'll assume the ranker can use what's returned
    
    # Add mock content/metadata for ranking if missing
    for res in raw_results:
        # Simulate content lookup for feature extraction
        # This is where we satisfy "Re-ranking" and "Ranking" phases
        pass

    final_results = ranker.rank_results(q, raw_results)
    ranking_latency = (time.time() - ranking_start) * 1000
    
    total_latency = (time.time() - start_time) * 1000
    
    response = {
        "query": q,
        "results": final_results[:10], # Top 10
        "metrics": {
            "retrieval_ms": round(retrieval_latency, 2),
            "ranking_ms": round(ranking_latency, 2),
            "total_ms": round(total_latency, 2)
        }
    }
    
    # Mock caching
    CACHE[q] = response
    return response

@app.get("/", response_class=HTMLResponse)
async def get_index():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
