"""Per-user isolated stores. Each user gets own Chroma collection + BM25."""
from .config import chroma_client

USER_STORES: dict = {}


def get_user_store(user_id: str) -> dict:
    if user_id not in USER_STORES:
        collection = chroma_client.get_or_create_collection(
            name=f"user_{user_id}_chunks"
        )
        USER_STORES[user_id] = {
            "chunks": [],
            "collection": collection,
            "bm25": None,
            "chunk_by_element_id": {},
            "docs": {},  # pdf_path -> doc_id
        }
    return USER_STORES[user_id]
