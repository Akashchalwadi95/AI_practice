from fastapi import FastAPI, Depends, status
from pydantic import BaseModel, Field
from sqlalchemy import select, String, Index, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from pgvector.sqlalchemy import Vector
from typing import AsyncGenerator
from contextlib import asynccontextmanager
from sentence_transformers import SentenceTransformer
import asyncio

DATABASE_URL = "postgresql+asyncpg://postgres:user@localhost:5432/postgres"

engine = create_async_engine(DATABASE_URL, echo=False)
async_session_factory = async_sessionmaker(engine, expire_on_commit=False)

class Base(DeclarativeBase):
    pass

class KnowledgeChunk(Base):
    __tablename__ = "chunk_storage_384"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    chunk_text: Mapped[str] = mapped_column(String(1000))
    source_file: Mapped[str] = mapped_column(String(50))
    embedding: Mapped[list[float]] = mapped_column(Vector(384))

    __table_args__ = (
        Index(
            "hnsw_embedding_idx_384",
            embedding,
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"}
        ),
    )


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    async with async_session_factory() as session:
        yield session


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield

    await engine.dispose()


class SearchQuery(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=3, ge=1, le=10)
    max_distance: float = Field(default=0.5)       

class SearchHit(BaseModel):
    id: int
    chunk_text: str
    source_file: str
    distance: float

class ChunkCreate(BaseModel):
    chunk_text: str = Field(..., min_length=5)
    source_file: str = Field(..., min_length=1)

class ChunkRead(BaseModel):
    id: int
    chunk_text: str
    source_file: str
    embedding: list[float]    

model = SentenceTransformer("all-MiniLM-L6-v2")

async def generate_embedding(text:str) -> list[float]:
    loop = asyncio.get_running_loop()
    embedding_array = await loop.run_in_executor(
        None, lambda: model.encode(text, convert_to_numpy=True)
    )
    return embedding_array.tolist()

app = FastAPI(title="Vector Search", lifespan=lifespan)

@app.post("/v1/chunks", response_model=ChunkRead, status_code=status.HTTP_201_CREATED)
async def create_chunk(
    payload: ChunkCreate,
    db: AsyncSession = Depends(get_db_session)):

    embedding = await generate_embedding(payload.chunk_text)

    new_chunk = KnowledgeChunk(
        chunk_text=payload.chunk_text,
        source_file=payload.source_file,
        embedding=embedding
    )

    db.add(new_chunk)
    await db.commit()
    await db.refresh(new_chunk)
    return new_chunk 

@app.post("/v1/chunks/search", response_model=list[SearchHit])
async def search_similar_chunks(
    payload: SearchQuery,
    db: AsyncSession = Depends(get_db_session)):
    
    query_embedding = await generate_embedding(payload.query)
    distance_col = KnowledgeChunk.embedding.cosine_distance(query_embedding)

    stmt = (
        select (
            KnowledgeChunk.id,
            KnowledgeChunk.chunk_text,
            KnowledgeChunk.source_file,
            distance_col.label("distance")
        )
    .where(distance_col <= payload.max_distance)
    .order_by(distance_col)
    .limit(payload.top_k
    )
    )

    await db.execute(text("SET LOCAL hnsw.ef_search = 40"))
    result = await db.execute(stmt)
    rows = result.all()

    return [
        SearchHit(
            id=row.id,
            chunk_text=row.chunk_text,
            source_file=row.source_file,
            distance=row.distance
        )
        for row in rows
    ]

