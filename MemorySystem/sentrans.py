from fastapi import FastAPI, Body
from sentence_transformers import SentenceTransformer
import numpy as np
import uvicorn

app = FastAPI()
model = None

@app.on_event("startup")
def load_model():
    global model
    print("Loading sentence transformer model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("Model loaded successfully")

@app.post("/embed")
async def get_embedding(text: str = Body(..., embed=True)):

    embedding = model.encode([text], convert_to_numpy=True)[0]
    return {"embedding": embedding.tolist()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8008)