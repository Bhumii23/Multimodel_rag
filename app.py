"""FastAPI entrypoint: upload PDFs per user, ask questions, check stats."""
import shutil
import os
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from src.ingestion import ingest_pdfs_for_user, sanity_check_user
from src.generation import ask
from src.store import get_user_store

app = FastAPI(title="Multimodal RAG API", version="2.0")
UPLOAD_DIR = "./uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.get("/")
def root():
    return {"status": "ok", "message": "Multimodal RAG API. POST /upload, POST /ask, GET /stats/{user_id}"}


@app.post("/upload")
async def upload(user_id: str = Form(...), files: list[UploadFile] = File(...)):
    paths = []
    for f in files:
        dest = os.path.join(UPLOAD_DIR, f"{user_id}_{f.filename}")
        with open(dest, "wb") as out:
            shutil.copyfileobj(f.file, out)
        paths.append(dest)
    store = ingest_pdfs_for_user(user_id, paths)
    return {"user_id": user_id, "chunks": len(store["chunks"]), "docs": list(store["docs"].keys())}


@app.post("/ask")
async def ask_q(user_id: str = Form(...), query: str = Form(...), top_k: int = Form(5)):
    result = ask(user_id, query, top_k=top_k)
    return JSONResponse(result)


@app.get("/stats/{user_id}")
def stats(user_id: str):
    store = get_user_store(user_id)
    return {"user_id": user_id, "chunks": len(store["chunks"]), "docs": list(store["docs"].keys())}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
