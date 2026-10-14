import asyncio
from typing import AsyncGenerator
from fastapi import FastAPI, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import String, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# --- 1. Database Setup ---
DATABASE_URL = "sqlite+aiosqlite:///./test_ai_logs.db"

engine = create_async_engine(DATABASE_URL, echo=False)
async_session_factory = async_sessionmaker(engine, expire_on_commit=False)

class Base(DeclarativeBase):
    pass

# --- 2. Database Model ---
class PromptLog(Base):
    __tablename__ = "prompt_logs"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    prompt: Mapped[str] = mapped_column(String(500))
    response: Mapped[str] = mapped_column(String(1000))
    model_name: Mapped[str] = mapped_column(String(50))

# --- 3. Dependency for DB Session Management ---
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    async with async_session_factory() as session:
        yield session

# --- 4. FastAPI Setup ---
app = FastAPI(title="Async DB Persistence API")

@app.on_event("startup")
async def init_tables():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

# Schemas
class LogCreate(BaseModel):
    prompt: str = Field(..., min_length=3)
    response: str
    model_name: str

class LogRead(BaseModel):
    id: int
    prompt: str
    response: str
    model_name: str

# --- 5. Routes ---
@app.post("/v1/logs", response_model=LogRead, status_code=status.HTTP_201_CREATED)
async def create_log(
    payload: LogCreate,
    db: AsyncSession = Depends(get_db_session)
):
    new_log = PromptLog(
        prompt=payload.prompt,
        response=payload.response,
        model_name=payload.model_name
    )
    db.add(new_log)
    await db.commit()
    await db.refresh(new_log)  # Reloads model to fetch generated auto-increment ID
    return new_log

@app.get("/v1/logs", response_model=list[LogRead])
async def list_logs(db: AsyncSession = Depends(get_db_session)):
    stmt = select(PromptLog)
    result = await db.execute(stmt)
    logs = result.scalars().all()
    return logs