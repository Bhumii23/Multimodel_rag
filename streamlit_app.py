"""Streamlit frontend: upload PDFs per user, ask questions, view sources."""
import os
import streamlit as st

from src.ingestion import ingest_pdfs_for_user
from src.generation import ask
from src.store import get_user_store

st.set_page_config(page_title="Multimodal RAG", layout="wide")
st.title("Multimodal RAG — PDFs with tables + images")

UPLOAD_DIR = "./uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ---- Sidebar: user + upload ----
with st.sidebar:
    st.header("1. User & PDFs")
    user_id = st.text_input("user_id", value="user_001")
    uploaded = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
    if st.button("Ingest PDFs", disabled=not uploaded):
        paths = []
        for f in uploaded:
            dest = os.path.join(UPLOAD_DIR, f"{user_id}_{f.name}")
            with open(dest, "wb") as out:
                out.write(f.getbuffer())
            paths.append(dest)
        with st.spinner("Parsing + chunking + indexing..."):
            store = ingest_pdfs_for_user(user_id, paths)
        st.success(f"Indexed {len(store['chunks'])} chunks from {len(store['docs'])} docs")

    if user_id:
        store = get_user_store(user_id)
        st.caption(f"Chunks: {len(store['chunks'])} | Docs: {len(store['docs'])}")
        for d in store["docs"]:
            st.code(os.path.basename(d))

# ---- Main: ask ----
st.header("2. Ask")
query = st.text_area("Question", placeholder="e.g. What do the red and blue regions indicate?")
top_k = st.slider("top_k (final sources)", 1, 10, 5)

if st.button("Ask", type="primary", disabled=not query or not user_id):
    store = get_user_store(user_id)
    if not store["chunks"]:
        st.warning("No chunks indexed for this user. Upload + Ingest PDFs first.")
        st.stop()
    with st.spinner("Hybrid search → rerank → expand → generate..."):
        result = ask(user_id, query, top_k=top_k)

    st.subheader("Answer")
    st.write(result["answer"])

    st.subheader("Sources")
    for s in result["sources"]:
        st.markdown(f"- `{os.path.basename(str(s['doc']))}` | {s['section']} (p{s['page']}, {s['type']})")

    with st.expander("Show retrieved context"):
        st.text(result["context"][:10000])
