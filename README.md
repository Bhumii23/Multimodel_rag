Multimodal RAG Pipeline

A Retrieval-Augmented Generation (RAG) system that answers questions over arbitrary PDF documents — including their text, tables, and images — with hierarchy-aware chunking and source-traceable answers.

Built and run end-to-end in Google Colab.



Overview

Most RAG pipelines treat a PDF as a flat wall of text. This one doesn't:

Structure-aware chunking — parses the document tree once and tracks nested headings (Section → Subsection → Sub-subsection) using a level-based stack, so every chunk knows its exact place in the document, not just "the last heading seen."
Tables handled row-by-row, each row tagged with its correct parent section — not just dumped as raw text.
Images captioned with Gemini and indexed as searchable text, so image content is retrievable through the same pipeline as everything else — no separate embedding model or vector space needed.
Hybrid retrieval — combines vector similarity search (semantic) with BM25 (keyword), then reranks candidates with a cross-encoder for precision.
Structure-aware context expansion — pulls in neighboring chunks from the same section (for text) or sibling rows from the same table (for table hits), so the LLM gets coherent context instead of isolated fragments.
Every answer comes with its source sections and page numbers, so results are traceable back to the original document.




  
Tech Stack
Tool	Purpose
Docling	Parses PDF structure — headings, tables, images
Sentence-Transformers (all-MiniLM-L6-v2)	Text embeddings for semantic search
ChromaDB	Vector database for embedding storage & retrieval
rank-bm25	Keyword-based (BM25) retrieval
Cross-Encoder (ms-marco-MiniLM-L-6-v2)	Reranks retrieved candidates
Groq (LLaMA 3.1)	Generates the final answer
Gemini	Generates captions for images found in the PDF
How It Works
Parse — Docling reads the PDF and identifies headings, paragraphs, tables, and images.
Chunk — the document tree is walked once. A heading stack tracks nesting depth, so every chunk (text, table row, image caption) is tagged with its full section path and page number.
Index — all chunks are embedded and stored in ChromaDB, and also indexed with BM25 for keyword matching.
Retrieve — a query triggers hybrid search (vector + keyword), and the top candidates are reranked with a cross-encoder.
Expand context — matching chunks are expanded with neighboring context from the same section, or sibling rows from the same table.
Answer — the assembled context is passed to the LLM, which answers strictly from the retrieved content, returning both the answer and its sources.
Setup







This notebook is designed to run in Google Colab.

Open the notebook in Colab.
Add the following secrets via Colab's Secrets panel (key icon in the left sidebar):
GROQ_API_KEY — from console.groq.com
GEMINI_API_KEY — from aistudio.google.com
Run all cells top to bottom. You'll be prompted to upload a PDF.
Use the ask("your question") function at the end to query the document.
