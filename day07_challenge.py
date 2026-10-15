from fastapi import FastAPI, status

app = FastAPI(title="docker contanerization")

@app.get("/v1/health", status_code=status.HTTP_200_OK)
async def getHealth():
    return{
        "status": "healthy",
        "service": "AI Engine Container"
    }