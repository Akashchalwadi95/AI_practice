from pydantic import BaseModel, Field
from typing import Annotated
from fastapi import FastAPI,Header, HTTPException, Depends, status

app = FastAPI(title="Rate limiting")

class User(BaseModel):
    username: str
    tier: str = Field(example="free or premium")

def get_current_user(x_user_id:Annotated[str|None, Header()] = None) -> User:
    if x_user_id == "user_free":
        return User(username="alice", tier="free")
    elif x_user_id == "user_vip":
        return User(username="bob", tier="premium")
    else:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="user not authenticated") 

def rate_limit_check(user: Annotated[User, Depends(get_current_user)]):
    if(user.tier == "free"):
        raise HTTPException(status_code=429, detail="Free tier rate limit exceeded. Please upgrade to premium.")
    else:
        return user    

@app.post("/v1/chat")
async def rate_limit(prompt: str, user:Annotated[User, Depends(rate_limit_check)]):
    return {
        "user": user.username,
        "status": "processed",
        "response": f"AI Answer to: {prompt}"
    }