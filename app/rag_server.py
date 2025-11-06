from fastapi import FastAPI
from pydantic import BaseModel
import chromadb
from chromadb.config import Settings
from openai import OpenAI

from .config import (
    PERSIST_DIRECTORY,
    COLLECTION_NAME,
    OPENAI_API_KEY,
    OPENAI_CHAT_MODEL,
    OPENAI_EMBED_MODEL,
)
from .ingest import build_index_if_empty

if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY")

# ---------- init ----------

build_index_if_empty()  # only runs if collection is empty

chroma = chromadb.Client(Settings(persist_directory=PERSIST_DIRECTORY))
collection = chroma.get_or_create_collection(COLLECTION_NAME)
client = OpenAI()

app = FastAPI(title="Raven RAG Bot")

SYSTEM_PROMPT = """
You are Raven Mott's RAG assistant.

Use ONLY the provided context, which comes from:
- Raven's websites,
- Raven's selected GitHub repositories,
- Raven's uploaded documents.

Rules:
- If the answer is not clearly supported by the context, say you don't know.
- Prefer concise, clear answers.
- When possible, reference where info came from (repo/file, URL, or doc name).
"""

# ---------- models ----------

class ChatRequest(BaseModel):
    message: str
    history: list[dict] | None = None   # [{role, content}]

class ChatResponse(BaseModel):
    answer: str
    sources: list[str]

# ---------- helpers ----------

def embed_query(q: str):
    resp = client.embeddings.create(
        model=OPENAI_EMBED_MODEL,
        input=[q]
    )
    return resp.data[0].embedding

def retrieve_context(query: str, k: int = 8):
    q_emb = embed_query(query)
    res = collection.query(
        query_embeddings=[q_emb],
        n_results=k,
    )

    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]

    blocks = []
    sources = []

    for d, m in zip(docs, metas):
        src = m.get("url") or m.get("filename") or m.get("repo") or m.get("source_type") or "unknown"
        tag = f"[{src}]"
        blocks.append(f"{tag} {d}")
        sources.append(src)

    # dedupe sources preserving order
    seen = set()
    uniq_sources = []
    for s in sources:
        if s not in seen:
            seen.add(s)
            uniq_sources.append(s)

    context_text = "\n\n".join(blocks)
    return context_text, uniq_sources

# ---------- routes ----------

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    context, sources = retrieve_context(req.message)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": f"Context:\n{context}"},
    ]

    if req.history:
        # include a short tail of history for style/continuity (no need for all)
        for m in req.history[-6:]:
            if m.get("role") in ("user", "assistant", "system"):
                messages.append({"role": m["role"], "content": m["content"]})

    messages.append({"role": "user", "content": req.message})

    completion = client.chat.completions.create(
        model=OPENAI_CHAT_MODEL,
        messages=messages,
        temperature=0.2,
    )

    answer = completion.choices[0].message.content.strip()
    return ChatResponse(answer=answer, sources=sources)

@app.get("/")
def health():
    return {"status": "ok", "collection": COLLECTION_NAME, "count": collection.count()}
