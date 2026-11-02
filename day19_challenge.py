from fastapi import FastAPI, Depends, status, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select, String, Index, text, func
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

class KeywordSearchHit(BaseModel):
    id: int
    chunk_text: str
    source_file: str
    keyword_rank: float  


class HybridSearchHit(BaseModel):
    id: int
    chunk_text: str
    source_file: str
    rrf_score: float
    distance: float | None = None
    keyword_rank: float | None = None
    

model = SentenceTransformer("all-MiniLM-L6-v2")

async def generate_embedding(text:str) -> list[float]:
    loop = asyncio.get_running_loop()
    embedding_array = await loop.run_in_executor(
        None, lambda: model.encode(text, convert_to_numpy=True)
    )
    return embedding_array.tolist()

def reciprocal_rank_fusion(
    vector_results: list[dict],
    keyword_results: list[dict],
    k: int = 60,
) -> list[dict]:
    """
    Combine vector and keyword rankings using RRF.

    Documents appearing in both lists accumulate scores
    from both rankings.
    """

    rrf_scores: dict[int, float] = {}
    doc_map: dict[int, dict] = {}

    # Process vector-search results.
    for rank, doc in enumerate(vector_results, start=1):
        doc_id = doc["id"]

        doc_map[doc_id] = {
            **doc_map.get(doc_id, {}),
            **doc,
        }

        rrf_scores[doc_id] = (
            rrf_scores.get(doc_id, 0.0)
            + 1.0 / (k + rank)
        )

        # Process keyword-search results.
    for rank, doc in enumerate(keyword_results, start=1):
        doc_id = doc["id"]

        doc_map[doc_id] = {
            **doc_map.get(doc_id, {}),
            **doc,
        }

        rrf_scores[doc_id] = (
            rrf_scores.get(doc_id, 0.0)
            + 1.0 / (k + rank)
        )
    
    # Sort by combined RRF score, highest first.
    sorted_doc_ids = sorted(
        rrf_scores,
        key=lambda doc_id: rrf_scores[doc_id],
        reverse=True,
    )

    fused_results = []

    for doc_id in sorted_doc_ids:
        fused_results.append({
            **doc_map[doc_id],
            "rrf_score": rrf_scores[doc_id],
        })

    return fused_results


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
    rows = result.mappings().all()

    return [
        SearchHit(**dict(row))
        for row in rows
    ]

@app.post("/v1/chunks/keyword-search", response_model=list[KeywordSearchHit])
async def keyword_search_chunks(
    payload: SearchQuery,
    db: AsyncSession = Depends(get_db_session)):

    search_vector = func.to_tsvector("english", KnowledgeChunk.chunk_text)

    search_query = func.websearch_to_tsquery("english", payload.query)

    # Rank keyword matches by PostgreSQL relevance.
    keyword_rank = func.ts_rank(
        search_vector,
        search_query,
    )   

    stmt = (
        select(
            KnowledgeChunk.id,
            KnowledgeChunk.chunk_text,
            KnowledgeChunk.source_file,
            keyword_rank.label("keyword_rank"),
        )
        .where(search_vector.bool_op("@@")(search_query))
        .order_by(keyword_rank.desc())
        .limit(payload.top_k)
    )

    result = await db.execute(stmt)
    rows = result.mappings().all()

    return [
        KeywordSearchHit(**dict(row))
        for row in rows
    ]

    
@app.post(
    "/v1/chunks/hybrid-search",
    response_model=list[HybridSearchHit],
)
async def hybrid_search_chunks(
    payload: SearchQuery,
    db: AsyncSession = Depends(get_db_session),
):
    # Generate the embedding only once.
    query_embedding = await generate_embedding(
        payload.query
    )

    # ------------------------------
    # A. Build vector-search query
    # ------------------------------

    distance_col = KnowledgeChunk.embedding.cosine_distance(
        query_embedding
    )

    vector_stmt = (
        select(
            KnowledgeChunk.id,
            KnowledgeChunk.chunk_text,
            KnowledgeChunk.source_file,
            distance_col.label("distance"),
        )
        .where(distance_col <= payload.max_distance)
        .order_by(distance_col)
        .limit(payload.top_k)
    )

    # ------------------------------
    # B. Build keyword-search query
    # ------------------------------

    search_vector = func.to_tsvector(
        "english",
        KnowledgeChunk.chunk_text,
    )

    search_query = func.websearch_to_tsquery(
        "english",
        payload.query,
    )

    keyword_rank = func.ts_rank(
        search_vector,
        search_query,
    )

    keyword_stmt = (
        select(
            KnowledgeChunk.id,
            KnowledgeChunk.chunk_text,
            KnowledgeChunk.source_file,
            keyword_rank.label("keyword_rank"),
        )
        .where(search_vector.bool_op("@@")(search_query))
        .order_by(keyword_rank.desc())
        .limit(payload.top_k)
    )

    # ------------------------------
    # C. Execute both searches
    # ------------------------------

    async with db.begin():
        await db.execute(
            text("SET LOCAL hnsw.ef_search = 40")
        )

        vector_result = await db.execute(vector_stmt)
        keyword_result = await db.execute(keyword_stmt)

        vector_hits = [
            dict(row)
            for row in vector_result.mappings().all()
        ]

        keyword_hits = [
            dict(row)
            for row in keyword_result.mappings().all()
        ]

    # ------------------------------
    # D. Fuse rankings with RRF
    # ------------------------------

    fused_results = reciprocal_rank_fusion(
        vector_results=vector_hits,
        keyword_results=keyword_hits,
        k=60,
    )


    # ------------------------------
    # E. Return the final top K
    # ------------------------------

    return [
        HybridSearchHit(**doc)
        for doc in fused_results[:payload.top_k]
    ]

    
@app.get(
    "/v1/chunks/{chunk_id}",
    response_model=ChunkRead,
)
async def get_chunk(
    chunk_id: int,
    db: AsyncSession = Depends(get_db_session),
):
    result = await db.execute(
        select(KnowledgeChunk).where(
            KnowledgeChunk.id == chunk_id
        )
    )

    chunk = result.scalar_one_or_none()

    if chunk is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Knowledge chunk not found",
        )

    return chunk