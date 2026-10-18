from fastapi import FastAPI, status, BackgroundTasks
from pydantic import BaseModel, Field
import asyncio

app = FastAPI(title="cached embeddings")

class EmbeddingTaskRequest(BaseModel):
    text: str = Field(..., min_length=5)

class EmbeddingTaskResponse(BaseModel):
    text : str
    vector: list[float]
    cached: bool

async def notify_vector_db_sync(text: str):
    await asyncio.sleep(1.0)
    print(f"[WEBHOOK] Synchronized vector index for text: '{text}'")

async def generateDummyVector(text:str):
    await asyncio.sleep(2.0)
    return [0.25, 0.5, 0.75, 1.0]  

vector_cache: dict[str, list[float]] = {}

@app.post("/v1/embed", response_model=EmbeddingTaskResponse, status_code=status.HTTP_200_OK)
async def process_query(payload: EmbeddingTaskRequest, bg_tasks:BackgroundTasks):
    if payload.text in vector_cache:
        return EmbeddingTaskResponse(
            text=payload.text,
            vector=vector_cache[payload.text],
            cached=True
        )
    result = await generateDummyVector(payload.text)
    vector_cache[payload.text] = result
    bg_tasks.add_task(notify_vector_db_sync, payload.text)

    return EmbeddingTaskResponse(
        text=payload.text,
        vector=result,
        cached=False
    )

