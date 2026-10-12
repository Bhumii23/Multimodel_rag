"""Hybrid search + rerank + context expansion."""
import numpy as np
from langfuse import observe
from .config import embedding_model, reranker
from .store import get_user_store


def min_max_normalize(scores):
    scores = np.array(scores, dtype=float)
    if len(scores) == 0:
        return scores
    lo, hi = scores.min(), scores.max()
    if hi == lo:
        return np.ones_like(scores)
    return (scores - lo) / (hi - lo)


@observe(name="hybrid_search")
def hybrid_search(user_id: str, query: str, top_k: int = 20, vector_weight: float = 0.7, keyword_weight: float = 0.3):
    store = get_user_store(user_id)
    chunks_u, collection, bm25 = store["chunks"], store["collection"], store["bm25"]
    n_results = min(top_k, len(chunks_u))
    if n_results == 0:
        return []
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)[0]
    chroma_results = collection.query(query_embeddings=[query_embedding.tolist()], n_results=n_results)
    vector_scores = {cid: 1 / (1 + dist) for cid, dist in zip(chroma_results["ids"][0], chroma_results["distances"][0])}
    normalized_bm25 = min_max_normalize(bm25.get_scores(query.lower().split()))

    combined = []
    for idx, chunk in enumerate(chunks_u):
        cid = f"chunk_{chunk['id']}"
        score = vector_weight * vector_scores.get(cid, 0.0) + keyword_weight * normalized_bm25[idx]
        combined.append({
            "chunk_id": cid, "element_id": chunk["element_id"],
            "text": chunk["text"], "metadata": chunk["metadata"], "score": float(score),
        })
    combined.sort(key=lambda x: x["score"], reverse=True)
    return combined[:top_k]


@observe(name="rerank_results")
def rerank_results(query: str, results: list, top_k: int = 5):
    if not results:
        return []
    scores = reranker.predict([[query, r["text"]] for r in results])
    for r, s in zip(results, scores):
        r["rerank_score"] = float(s)
    results.sort(key=lambda x: x["rerank_score"], reverse=True)
    return results[:top_k]


def expand_context(user_id: str, reranked_results: list, window_size: int = 2, max_table_siblings: int = 8):
    store = get_user_store(user_id)
    chunks_u, chunk_by_element_id = store["chunks"], store["chunk_by_element_id"]
    expanded, seen_ids = [], set()
    for hit in reranked_results:
        group = [hit]
        seen_ids.add(hit["chunk_id"])
        if hit["metadata"]["type"] == "table_row":
            table_id, doc_id = hit["metadata"]["table_id"], hit["metadata"].get("doc_id")
            siblings = [c for c in chunks_u
                        if c["metadata"]["table_id"] == table_id
                        and c["metadata"].get("doc_id") == doc_id
                        and f"chunk_{c['id']}" != hit["chunk_id"]][:max_table_siblings]
            for s in siblings:
                cid = f"chunk_{s['id']}"
                if cid not in seen_ids:
                    group.append({"chunk_id": cid, "element_id": s["element_id"], "text": s["text"], "metadata": s["metadata"]})
                    seen_ids.add(cid)
        else:
            for offset in range(-window_size, window_size + 1):
                neighbor = chunk_by_element_id.get(hit["element_id"] + offset)
                if neighbor is None:
                    continue
                if neighbor["metadata"]["section"] != hit["metadata"]["section"]:
                    continue
                if neighbor["metadata"].get("doc_id") != hit["metadata"].get("doc_id"):
                    continue
                cid = f"chunk_{neighbor['id']}"
                if cid not in seen_ids:
                    group.append({"chunk_id": cid, "element_id": neighbor["element_id"], "text": neighbor["text"], "metadata": neighbor["metadata"]})
                    seen_ids.add(cid)
        group.sort(key=lambda x: x["element_id"])
        expanded.append({"retrieved": hit, "context_chunks": group})
    return expanded


def assemble_context(expanded_results: list, max_chars: int = 10000) -> str:
    seen, parts = set(), []
    for group in expanded_results:
        for c in group["context_chunks"]:
            if c["text"] in seen:
                continue
            seen.add(c["text"])
            meta = c["metadata"]
            parts.append(f"[Doc: {meta.get('doc_name','?')} | Section: {meta['section']} | Page: {meta['page']}]\n{c['text']}")
    context = "\n\n".join(parts)
    return context[:max_chars] + "\n[... context truncated ...]" if len(context) > max_chars else context
