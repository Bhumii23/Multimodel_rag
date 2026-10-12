"""Index chunks into ChromaDB + BM25."""
import uuid
from collections import Counter
from langfuse import observe
from rank_bm25 import BM25Okapi
from .config import embedding_model
from .store import get_user_store
from .parsing import parse_pdf
from .chunking import build_document_index


def prepare_metadata(metadata: dict) -> dict:
    clean = {}
    for k, v in metadata.items():
        if v is None:
            continue
        clean[k] = v if isinstance(v, (str, int, float, bool)) else str(v)
    return clean


def sanity_check_user(user_id: str, n: int = 5):
    store = get_user_store(user_id)
    chunks = store["chunks"]
    print(f"User {user_id}: {len(chunks)} chunks | docs: {list(store['docs'].keys())}")
    print(Counter(c["metadata"]["type"] for c in chunks))
    for c in chunks[:n]:
        print(f"[{c['metadata']['type']}] {c['metadata']['doc_name']} | {c['metadata']['section']} | p{c['metadata']['page']}")
        print(c["text"][:120], "\n")


@observe(name="ingest_pdfs_for_user")
def ingest_pdfs_for_user(user_id: str, pdf_paths: list):
    store = get_user_store(user_id)
    added_new_pdf = False

    for pdf_path in pdf_paths:
        if pdf_path in store["docs"]:
            print(f"Skipping already ingested {pdf_path} for {user_id}")
            continue
        doc_id = f"{pdf_path}_{uuid.uuid4().hex[:8]}"
        doc = parse_pdf(pdf_path)
        new_chunks = build_document_index(doc, user_id, doc_id, pdf_path)
        store["chunks"].extend(new_chunks)
        store["docs"][pdf_path] = doc_id
        added_new_pdf = True
        print(f"{pdf_path}: {len(new_chunks)} chunks")

    if not store["chunks"]:
        print("No chunks available to index.")
        return store
    if not added_new_pdf and store["bm25"] is not None:
        print("No new PDFs; existing indices retained.")
        return store

    chunks_u = store["chunks"]
    texts = [c["text"] for c in chunks_u]
    ids = [f"chunk_{c['id']}" for c in chunks_u]
    metadatas = [prepare_metadata(c["metadata"]) for c in chunks_u]

    embeddings = embedding_model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    try:
        existing_ids = store["collection"].get()["ids"]
        if existing_ids:
            store["collection"].delete(ids=existing_ids)
    except Exception as e:
        print("Warning while clearing ChromaDB:", e)

    store["collection"].add(ids=ids, documents=texts, embeddings=embeddings.tolist(), metadatas=metadatas)
    store["bm25"] = BM25Okapi([t.lower().split() for t in texts])
    store["chunk_by_element_id"] = {
        c["element_id"]: c for c in chunks_u if c["metadata"]["type"] != "table_row"
    }
    print(f"User {user_id}: indexed {store['collection'].count()} chunks from {len(store['docs'])} PDFs.")
    return store
