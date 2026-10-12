"""Central config: env, models, clients. Import this, don't duplicate it."""
import os
from dotenv import load_dotenv

load_dotenv()

from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_groq import ChatGroq
from groq import Groq
from langfuse import get_client
import chromadb

# ---- Keys (replaces google.colab userdata) ----
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    raise ValueError("Set GROQ_API_KEY in .env (see .env.example)")

os.environ["GROQ_API_KEY"] = GROQ_API_KEY
os.environ["LANGFUSE_PUBLIC_KEY"] = os.getenv("LANGFUSE_PUBLIC_KEY", "")
os.environ["LANGFUSE_SECRET_KEY"] = os.getenv("LANGFUSE_SECRET_KEY", "")
os.environ["LANGFUSE_HOST"] = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

# ---- Models ----
TEXT_MODEL = os.getenv("TEXT_MODEL", "openai/gpt-oss-20b")
VISION_MODEL = os.getenv("VISION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")

embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

llm = ChatGroq(model=TEXT_MODEL)
vision_client = Groq(api_key=GROQ_API_KEY)

langfuse = get_client()
chroma_client = chromadb.PersistentClient(path=os.getenv("CHROMA_PATH", "./chroma_db"))
