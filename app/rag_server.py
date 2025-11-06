import json
import math

from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI

from .config import (
    OPENAI_API_KEY,
    OPENAI_CHAT_MODEL,
    OPENAI_EMBED_MODEL,
)
from .ingest import build_index_if_missing, INDEX_PATH

if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY")

client = OpenAI()

app = FastAPI(title="Raven RAG Bot (Lite)")

# Build index on first boot if missing
build_index_if_missing()

# Load index into memory
with INDEX_PATH.open("r") as f:
    INDEX = json.load(f)

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


class ChatRequest(BaseModel):
    message: str
    history: list[dict] | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]


def embed_query(text: str):
    resp = client.embeddings.create(
        model=OPENAI_EMBED_MODEL,
        input=[text],
    )
    return resp.data[0].embedding


def cosine(a, b):
    s = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        s += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return s / (na ** 0.5 * nb ** 0.5)


def retrieve(query: str, k: int = 8):
    q_emb = embed_query(query)
    scored = []
    for entry in INDEX:
        sim = cosine(q_emb, entry["embedding"])
        scored.append((sim, entry))
    scored.sort(key=lambda t: t[0], reverse=True)
    top = [e for (s, e) in scored[:k] if s > 0]

    blocks = []
    srcs = []
    for e in top:
        meta = e.get("meta", {})
        src = (
            meta.get("url")
            or meta.get("filename")
            or meta.get("repo")
            or meta.get("source_type")
            or "unknown"
        )
        tag = f"[{src}]"
        blocks.append(f"{tag} {e['text']}")
        srcs.append(src)

    # dedupe sources
    seen = set()
    uniq = []
    for s in srcs:
        if s not in seen:
            seen.add(s)
            uniq.append(s)

    context = "\n\n".join(blocks)
    return context, uniq


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    context, sources = retrieve(req.message)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": f"Context:\n{context}"},
    ]

    if req.history:
        for m in req.history[-6:]:
            if m.get("role") in ("user", "assistant", "system") and m.get("content"):
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
def root():
    return {"status": "ok", "entries": len(INDEX)}
