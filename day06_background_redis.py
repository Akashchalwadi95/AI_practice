import asyncio
from fastapi import FastAPI, BackgroundTasks, status
from pydantic import BaseModel, Field

app = FastAPI(title="Async Background Processing & Caching")

# Simulated in-memory cache store (Simulates Redis key-value with TTL)
simulated_redis_cache: dict[str, str] = {}

class QueryRequest(BaseModel):
    prompt: str = Field(..., min_length=3)

class QueryResponse(BaseModel):
    prompt: str
    response: str
    cached: bool

# --- Background Task Function ---
async def log_analytics_bg(prompt: str, response: str):
    """
    Simulates sending analytics metrics to a monitoring system 
    WITHOUT blocking the client's HTTP response.
    """
    await asyncio.sleep(2.0)  # Heavy background I/O
    print(f" [BACKGROUND TASK] Logged analytics for prompt: '{prompt[:15]}...'")

# --- Heavy LLM Computation Function ---
async def heavy_llm_computation(prompt: str) -> str:
    await asyncio.sleep(3.0)  # Simulates 3-second LLM API delay
    return f"AI Generated response for: '{prompt}'"

# --- Route with Cache-Aside & Background Task ---
@app.post("/v1/query", response_model=QueryResponse, status_code=status.HTTP_200_OK)
async def process_query(
    payload: QueryRequest,
    bg_tasks: BackgroundTasks
):
    cache_key = f"cache:{payload.prompt}"

    # 1. Check Cache Hit
    if cache_key in simulated_redis_cache:
        print(f" [CACHE HIT] Returning instant response for '{payload.prompt}'")
        return QueryResponse(
            prompt=payload.prompt,
            response=simulated_redis_cache[cache_key],
            cached=True
        )

    # 2. Cache Miss: Execute LLM Call
    print(f" [CACHE MISS] Computing heavy response for '{payload.prompt}'...")
    llm_output = await heavy_llm_computation(payload.prompt)

    # 3. Store in Cache
    simulated_redis_cache[cache_key] = llm_output

    # 4. Schedule non-blocking background task
    bg_tasks.add_task(log_analytics_bg, payload.prompt, llm_output)

    # 5. Return HTTP Response immediately
    return QueryResponse(
        prompt=payload.prompt,
        response=llm_output,
        cached=False
    )