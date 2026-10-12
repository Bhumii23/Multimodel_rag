# Multimodal RAG System — `multimodel_rag_clean_finall.py`

Retrieval-Augmented Generation pipeline for PDFs with text, tables, headings, nested sections, and images.

**Stack:** Docling + ChromaDB + BM25 + Cross-Encoder Reranker + Groq LLM + Langfuse

## 1. Workflow Overview

```
PDFs → Docling Parse → Structure-Aware Chunking → ChromaDB + BM25 Index → Hybrid Search → Rerank → Context Expansion → Groq Answer
                                                                                        ↘ Langfuse (observability only)
```

## 2. Step-by-Step

### Step 1 — Setup
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Text LLM: Groq `openai/gpt-oss-20b` via `ChatGroq`
- Vision: Groq `meta-llama/llama-4-scout-17b-16e-instruct` via `Groq()` for image captions
- Keys from Colab Secrets: `GROQ_API_KEY`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`

### Step 2 — Upload & Parse PDFs (per user)
- `current_user_id` isolates users. Each user gets own Chroma collection `user_{id}_chunks` via `get_user_store(user_id)`.
- `files.upload()` → `pdf_paths`
- `parse_pdf(pdf_path)` — `@observe(name="parse_pdf")`, uses `DocumentConverter` with `PdfPipelineOptions(images_scale=2.0, generate_page_images=True, generate_picture_images=True)`.

### Step 3 — Structure-Aware Chunking
Single pass in `build_document_index(doc, user_id, doc_id, doc_name)` — `@observe(name="build_document_index")`:
- `Section` class + `heading_stack` tracks full breadcrumb `Section > Subsection`, not just last heading.
- `classify(item)`: heading / table / picture / list_item / text using `label` + class name.
- `get_page(item)`: from `item.prov[0].page_no`.
- Heading → indexed as `type: heading` chunk.
- Table → `export_to_dataframe()` → one chunk per row: `Table (Section: ...): col: val | ...`, `type: table_row`, with `table_id`, `row_index`.
- Picture → `caption_image(item, doc, section)` — `@observe(name="caption_image")`, sends PNG to vision model, returns 1-3 sentence caption, indexed as `Image (Section: ...): caption`, `type: image_caption`.
- Text/list → indexed if `len(text) >= 5`, with `section`, `page`.
- Every chunk tagged with `user_id`, `doc_id`, `doc_name`.

### Step 4 — Indexing
`ingest_pdfs_for_user(user_id, pdf_paths)` — `@observe`:
1. Skip already-ingested PDFs (`store["docs"]`).
2. `parse_pdf` → `build_document_index` → `store["chunks"].extend(new_chunks)`.
3. Embed all texts with `embedding_model.encode()`, clear + `collection.add(ids, documents, embeddings, metadatas)`.
4. Build `BM25Okapi` over lowercased tokens.
5. Build `chunk_by_element_id` (excludes `table_row` for neighbour lookup).

`sanity_check_user()` prints counts by `type` and `doc_name`.

### Step 5 — Retrieval: Hybrid Search + Rerank
- `hybrid_search(user_id, query, top_k=20)` — `@observe`: vector search (Chroma, `1/(1+dist)`) + BM25, min-max normalized, weighted `0.7 * vector + 0.3 * keyword`, sorted top_k.
- `rerank_results(query, results, top_k=5)` — `@observe`: `CrossEncoder.predict([query, text])` → `rerank_score`, sorted top_k.

### Step 6 — Context Expansion & Generation
- `expand_context(user_id, reranked)`: table_row hits → pull sibling rows with same `table_id` + `doc_id` (max 8); other hits → pull neighbours `element_id ±2` with same `section` + `doc_id`.
- `assemble_context(expanded, max_chars=10000)`: dedup, format `[Doc: ... | Section: ... | Page: ...]\ntext`.
- `generate_answer(query, context)` — `@observe`: strict prompt `Answer ONLY from context, else say you don't know` → `llm.invoke(prompt).content`.
- `ask(user_id, query)` — `@observe(name="ask")`: orchestrates hybrid → rerank → expand → assemble → generate, returns `{answer, context, sources}` + `langfuse.score(name="retrieved_chunks", value=len(reranked))`.

## 3. What Langfuse Does (and Does NOT Do)

**Does:**
- Creates one trace for `ask`, with nested spans for `parse_pdf`, `build_document_index`, `caption_image`, `ingest_pdfs_for_user`, `hybrid_search`, `rerank_results`, `generate_answer`.
- Logs inputs/outputs, latency, and error status + stacktrace per span.
- Logs metric `retrieved_chunks` via `langfuse.score()`.
- Viewable in dashboard at `https://cloud.langfuse.com`.

**Does NOT:**
- Handle / retry / fix a chunking error.
- Respond to the user.
- If `build_document_index` throws an uncaught exception, Langfuse just marks that span as failed — your code still crashes.
- Table/image errors inside are `try/except + print + skip`, so they don't even appear as errors in Langfuse — that row/image is just skipped.

If you want resilience, add your own `try/except` around chunking/ingestion.

## 4. Run

```python
!pip install -q docling sentence-transformers rank-bm25 pandas chromadb langchain_groq google-generativeai langfuse
# set GROQ_API_KEY, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY in Colab Secrets
ingest_pdfs_for_user("user_001", pdf_paths)
result = ask("user_001", "your question")
print(result["answer"])
```

## 5. Files
- `multimodel_rag_clean_finall.py` — full pipeline (this README describes it)

