import os, json
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from .species import octopus, seahorse, dolphin

EMBED_MODEL = None
CHROMA_CLIENT = None
COLLECTION = None

def _get_embedder():
    global EMBED_MODEL
    if EMBED_MODEL is None:
        # Small, free model
        EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return EMBED_MODEL

def _get_chroma():
    global CHROMA_CLIENT, COLLECTION
    if CHROMA_CLIENT is None:
        CHROMA_CLIENT = chromadb.Client(Settings(persist_directory=".chroma"))
        COLLECTION = CHROMA_CLIENT.get_or_create_collection("cora_kb")
    return CHROMA_CLIENT, COLLECTION

def _seed_docs():
    # Load hard-coded species seeds (safe, high-level)
    docs = []
    for item in octopus.SEED + seahorse.SEED + dolphin.SEED:
        docs.append(item)
    # Optional: JSONL seed file
    seed_path = os.path.join(os.path.dirname(__file__), "data", "seed_documents.jsonl")
    if os.path.exists(seed_path):
        with open(seed_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    docs.append(json.loads(line))
                except:
                    pass
    return docs

def initialize_index():
    _get_embedder()
    _, col = _get_chroma()
    if col.count() > 0:
        return

    docs = _seed_docs()
    ids = [d["id"] for d in docs]
    texts = [
        f'{d.get("title","")} | {d.get("species","")} | {", ".join(d.get("tags",[]))} | {d.get("summary","")}'
        for d in docs
    ]
    metadatas = docs
    col.add(ids=ids, documents=texts, metadatas=metadatas)

def search(query: str, k: int = 5):
    _get_embedder()
    _, col = _get_chroma()
    results = col.query(query_texts=[query], n_results=k)
    items = []
    if results and results.get("ids"):
        for i in range(len(results["ids"][0])):
            items.append({
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "meta": results["metadatas"][0][i]
            })
    return items
