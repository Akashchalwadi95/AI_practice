from fastapi import FastAPI, status, Depends
from sqlalchemy.orm import mapped_column, Mapped, DeclarativeBase
import asyncio
from sqlalchemy import select, String
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from pydantic import BaseModel, Field
from typing import AsyncGenerator
from pgvector.sqlalchemy import Vector
from contextlib import asynccontextmanager

DATABASE_URL = "postgresql+asyncpg://postgres:user@localhost:5432/postgres"

engine = create_async_engine(DATABASE_URL, echo=False)
async_session_factory = async_sessionmaker(engine, expire_on_commit=False)

class Base(DeclarativeBase):
    pass

class KnowledgeChunk(Base):
    __tablename__ = "chunk_storage"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    chunk_text: Mapped[str] = mapped_column(String(1000))
    source_file: Mapped[str] = mapped_column(String(50))
    embedding: Mapped[list[float]] = mapped_column(Vector(4))


# --- DB Dependency ---
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    async with async_session_factory() as session:
        yield session

class ChunkCreate(BaseModel):
    chunk_text: str = Field(..., min_length=5)
    source_file: str = Field(..., min_length=1)
    embedding: list[float] = Field(..., min_length=4, max_length=4)

class ChunkRead(BaseModel):
    id: int
    chunk_text: str
    source_file: str
    embedding: list[float]    

# --- Modern FastAPI Lifespan ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield

    # Shutdown: dispose of the database connection pool
    await engine.dispose()

app = FastAPI(title="pgvector Knowledge Base API", lifespan=lifespan)

@app.get("/v1/chunks", response_model=list[ChunkRead])
async def list_chunks(db: AsyncSession = Depends(get_db_session)):
    stmt = select(KnowledgeChunk)
    result = await db.execute(stmt)
    chunks = result.scalars().all()
    return chunks

@app.post("/v1/chunks", response_model=ChunkRead, status_code=status.HTTP_201_CREATED)    
async def create_chunk(payload: ChunkCreate, db: AsyncSession = Depends(get_db_session)):
    new_chunk = KnowledgeChunk(
        chunk_text=payload.chunk_text,
        source_file=payload.source_file,
        embedding=payload.embedding
    )

    db.add(new_chunk)
    await db.commit()
    await db.refresh(new_chunk)
    return new_chunk