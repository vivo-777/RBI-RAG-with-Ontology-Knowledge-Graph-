from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import json
import config
import os
import time
from datetime import datetime

from generation.dspy_advisor import generate_dspy_answer
from generation.citation_verifier import verify_response

app = FastAPI(title="RBI Regulatory Circular RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    return {"status": "ok", "model": config.GROQ_DEFAULT_MODEL}

from pydantic import BaseModel, validator
import hashlib

_query_cache = {}
CACHE_TTL_SECONDS = 300  # 5 minutes

def get_cached_result(query: str):
    key = hashlib.md5(query.lower().strip().encode()).hexdigest()
    if key in _query_cache:
        result, timestamp = _query_cache[key]
        if time.time() - timestamp < CACHE_TTL_SECONDS:
            return result
        del _query_cache[key]
    return None

def set_cached_result(query: str, result: dict):
    key = hashlib.md5(query.lower().strip().encode()).hexdigest()
    _query_cache[key] = (result, time.time())

class QueryRequest(BaseModel):
    query: str

    @validator("query")
    def validate_query(cls, v):
        if not v or not v.strip():
            raise ValueError("Query must not be empty")
        if len(v) > 2000:
            raise ValueError("Query must be under 2000 characters")
        return v.strip()

def log_audit_trail(audit_data: dict):
    log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "audit_logs"))
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"query_log_{timestamp}.json")
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(audit_data, f, indent=2)

@app.post("/api/chat")
async def chat_endpoint(request: QueryRequest):
    query = request.query
    
    cached = get_cached_result(query)
    
    async def stream_generator():
        if cached:
            # Re-yield cached chunks
            citations = cached.get("citations", [])
            yield f"data: {json.dumps({'citations': citations})}\n\n"
            yield f"data: {json.dumps({'chunk': cached.get('final_verified_response', '')})}\n\n"
            yield f"data: {json.dumps(cached)}\n\n"
            return

        # Execute DSPy pipeline which handles decomposition, retrieval, and generation
        try:
            dspy_result = generate_dspy_answer(query)
            retrieved_chunks = dspy_result.get("retrieved_chunks", [])
            final_answer = dspy_result.get("answer", "")
            model_used = dspy_result.get("model", "unknown")
            latency = dspy_result.get("latency", 0.0)
            
            # Explicit abstention gate (if retrieved chunks is empty)
            if not retrieved_chunks:
                abstention_msg = "The provided regulatory context does not contain sufficient information to answer this query definitively."
                verification_result = {
                    "response": abstention_msg,
                    "accuracy": 0.0,
                    "claim_labels": []
                }
                final_answer = abstention_msg
            else:
                context_texts = [c["payload"].get("content", "") for c in retrieved_chunks]
                verification_result = verify_response(final_answer, context_texts)
                
                # Apply self-consistency voting logic:
                # If any claim is NOT present in at least 1 of the votes, demote it.
                votes = dspy_result.get("votes", [])
                if votes:
                    # simplistic check:
                    vote_text = " ".join(votes).lower()
                    for idx, claim_data in enumerate(verification_result["claim_labels"]):
                        if claim_data["label"] == "supported":
                            # Check if key nouns or numbers exist in votes
                            import re
                            numbers = set(re.findall(r'\b\d+(?:\.\d+)?\b', claim_data["claim"]))
                            for num in numbers:
                                if str(num) not in vote_text:
                                    claim_data["label"] = "partially_supported"
                                    # Update the final response text
                                    verification_result["response"] = verification_result["response"].replace(
                                        claim_data["claim"], 
                                        claim_data["claim"] + " [⚠️ Requires Analyst Review]"
                                    )
                                    break

                
            # Audit log data
            audit_entry = {
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "retrieved_doc_ids": [c["id"] for c in retrieved_chunks],
                "retrieved_spans": [c["payload"].get("content", "") for c in retrieved_chunks],
                "model_version": model_used,
                "latency_sec": latency,
                "verifier_labels": verification_result.get("claim_labels", []),
                "final_answer": verification_result["response"]
            }
            log_audit_trail(audit_entry)

            # We yield a single chunk since the underlying DSPy logic might not easily stream
            yield f"data: {json.dumps({'citations': retrieved_chunks})}\n\n"
            yield f"data: {json.dumps({'chunk': final_answer})}\n\n"
            final_res = {'done': True, 'final_verified_response': verification_result['response'], 'accuracy': verification_result['accuracy'], 'citations': retrieved_chunks}
            set_cached_result(query, final_res)
            yield f"data: {json.dumps(final_res)}\n\n"

        except Exception as e:
            import logging
            logging.error(f"Error in chat_endpoint: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            
    return StreamingResponse(stream_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    uvicorn.run(app, host=config.API_HOST, port=config.API_PORT)
