from fastapi import FastAPI

app = FastAPI(
    title="PML: JokeKook",
)

@app.get("/")
async def root():
    return {"message": "Hello World"}
