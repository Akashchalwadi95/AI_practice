from fastapi import FastAPI, Depends, status, HTTPException
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import String, select
from pydantic import BaseModel
import uuid
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

DATABASE_URL = "sqlite+aiosqlite:///./agent_sessions.db"

engine = create_async_engine(DATABASE_URL, echo=False)
async_session_factory = async_sessionmaker(engine, expire_on_commit=False)

class Base(DeclarativeBase):
    pass

class AgentSession(Base):
    __tablename__ = "sessionLogs"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(36), unique=True, )
    user_id: Mapped[str] = mapped_column(String(50))
    agent_type: Mapped[str] = mapped_column(String(100))
    message_count: Mapped[int] = mapped_column(default=0)

async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    async with async_session_factory() as session:
        yield session


class UserDetails(BaseModel):
    user_id: str
    agent_type: str    

class SessionId(BaseModel):
    session_id: str

class SessionRead(BaseModel):
    id: int
    session_id: str
    user_id: str
    agent_type: str
    message_count: int    

app = FastAPI(title="Agent Session")


@app.on_event("startup")
async def init_tables():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

@app.post("/v1/sessions", response_model=SessionRead, status_code=status.HTTP_201_CREATED)
async def saveSession(payload: UserDetails, db: AsyncSession = Depends(get_db_session)):
    new_Session = AgentSession(
        session_id= str(uuid.uuid4()),
        user_id= payload.user_id,
        agent_type= payload.agent_type,
        message_count= 0
    )

    db.add(new_Session)
    await db.commit()
    await db.refresh(new_Session)
    return new_Session


@app.get("/v1/sessions/{user_id}", response_model=list[SessionId])
async def getData(user_id:str ,db: AsyncSession = Depends(get_db_session)):
    stmt = select(AgentSession).where(AgentSession.user_id == user_id)
    result = await db.execute(stmt)
    session = result.scalars().all()
    return session