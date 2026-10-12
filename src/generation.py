"""Answer generation + top-level orchestrator."""
from langfuse import observe
from .config import llm, langfuse
from .retrieval import hybrid_search, rerank_results, expand_context, assemble_context


def create_rag_prompt(query: str, context: str) -> str:
    return (
        "Answer the question using ONLY the context below. "
        "If the answer is not in the context, say you don't know.\n\n"
        f"Context:\n{context}\n\nQuestion: {query}"
    )


@observe(name="generate_answer")
def generate_answer(query: str, context: str) -> str:
    return llm.invoke(create_rag_prompt(query, context)).content


@observe(name="ask")
def ask(user_id: str, query: str, top_k: int = 5) -> dict:
    candidates = hybrid_search(user_id, query, top_k=20)
    reranked = rerank_results(query, candidates, top_k=top_k)
    expanded = expand_context(user_id, reranked)
    context = assemble_context(expanded)
    answer = generate_answer(query, context)
    sources = [
        {"doc": r["metadata"].get("doc_name"), "section": r["metadata"]["section"],
         "page": r["metadata"]["page"], "type": r["metadata"]["type"]}
        for r in reranked
    ]
    try:
        langfuse.score(name="retrieved_chunks", value=len(reranked))
    except Exception as e:
        print(f"[Langfuse Warning] Could not log score: {e}")
    return {"user_id": user_id, "query": query, "answer": answer, "context": context, "sources": sources}
