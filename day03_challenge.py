from pydantic import BaseModel, Field
import asyncio
from fastapi import FastAPI, status, HTTPException

app = FastAPI(title="My Title", version="1.0.0")

class EmbeddingRequest(BaseModel):
    texts: list[str] = Field(min_length=1)
    dimensions: int = Field(default=1536, ge=0)

class EmbeddingResponse(BaseModel):
    total_texts: int
    dimensions: int
    embeddings: list[list[float]]   

async def mock_embed_text(text: str, dim: int) -> list[float]:
    await asyncio.sleep(0.5)
    return [0.1]*dim


@app.get("/")
async def root():
    return {"message": "LLM Gateway Service is Online"}


@app.post("/v1/embeddings", response_model=EmbeddingResponse, status_code=status.HTTP_200_OK)
async def processInput(payload: EmbeddingRequest):

    tasks = [mock_embed_text(text, payload.dimensions) for text in payload.texts]
    embeddings = await asyncio.gather(*tasks)

    return EmbeddingResponse(
        total_texts = len(payload.texts),
        dimensions = payload.dimensions,
        embeddings = embeddings
    )