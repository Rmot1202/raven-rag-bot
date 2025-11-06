import os
import re
from pathlib import Path
from urllib.parse import urljoin, urlparse

import chromadb
from chromadb.config import Settings
import requests
from bs4 import BeautifulSoup
from github import Github
from openai import OpenAI
from pypdf import PdfReader
import docx

from .config import (
    BASE_DIR,
    PERSIST_DIRECTORY,
    DOCS_DIR,
    OPENAI_API_KEY,
    OPENAI_EMBED_MODEL,
    GITHUB_TOKEN,
    GITHUB_USERNAME,
    GITHUB_REPOS_ALLOWLIST,
    SEED_SITES,
    MAX_WEB_PAGES,
    COLLECTION_NAME,
)

# ---------- setup ----------

if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY")

client = OpenAI()
chroma = chromadb.Client(Settings(persist_directory=PERSIST_DIRECTORY))
collection = chroma.get_or_create_collection(COLLECTION_NAME)

TEXT_EXT = {".md", ".txt"}
CODE_EXT = {
    ".py", ".ipynb", ".kt", ".java", ".c", ".cpp",
    ".js", ".ts", ".rs", ".go", ".rb", ".php"
}
SKIP_DIRS = {".git", "node_modules", "build", "dist", ".gradle", ".idea", "__pycache__"}

# ---------- shared helpers ----------

def clean_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def chunk(text: str, max_tokens=400, overlap=80):
    """
    Rough char-based chunking. 4 chars ≈ 1 token.
    """
    text = clean_ws(text)
    if not text:
        return []
    max_len = max_tokens * 4
    ov_len = overlap * 4
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + max_len, n)
        cut = text.rfind(". ", start, end)
        if cut == -1 or cut < start + int(max_len * 0.4):
            cut = end
        piece = text[start:cut].strip()
        if len(piece) > 60:
            chunks.append(piece)
        start = max(0, cut - ov_len)
    return chunks

def embed_batch(texts):
    resp = client.embeddings.create(
        model=OPENAI_EMBED_MODEL,
        input=texts
    )
    return [d.embedding for d in resp.data]

def add_chunks(chunks, base_meta, id_prefix):
    if not chunks:
        return
    embeddings = embed_batch(chunks)
    ids = [f"{id_prefix}-{i}" for i in range(len(chunks))]
    metas = [base_meta.copy() for _ in chunks]
    collection.add(
        ids=ids,
        embeddings=embeddings,
        metadatas=metas,
        documents=chunks,
    )
    print(f"  -> {id_prefix}: {len(chunks)} chunks")

# ---------- local docs ----------

def load_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)

def load_docx(path: Path) -> str:
    doc = docx.Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs)

def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")

def ingest_local_docs():
    if not DOCS_DIR.exists():
        print("[docs] no docs dir, skipping")
        return

    print(f"[docs] ingesting from {DOCS_DIR}")
    for path in DOCS_DIR.rglob("*"):
        if not path.is_file():
            continue
        ext = path.suffix.lower()
        try:
            if ext == ".pdf":
                raw = load_pdf(path)
            elif ext == ".docx":
                raw = load_docx(path)
            elif ext in TEXT_EXT:
                raw = load_text(path)
            else:
                continue
        except Exception as e:
            print(f"  !! failed {path}: {e}")
            continue

        if len(raw.strip()) < 80:
            continue

        label = str(path.relative_to(DOCS_DIR))
        text = f"{label}\n\n{raw}"
        chunks = chunk(text)
        meta = {
            "source_type": "local_doc",
            "filename": label,
        }
        add_chunks(chunks, meta, f"doc-{label.replace('/','_')}")

# ---------- website crawl ----------

def ingest_websites():
    if not SEED_SITES:
        print("[web] no SEED_SITES, skipping")
        return

    print("[web] crawling websites...")
    visited = set()

    for seed in SEED_SITES:
        domain = urlparse(seed).netloc
        queue = [seed]

        while queue and len(visited) < MAX_WEB_PAGES:
            url = queue.pop(0)
            if url in visited:
                continue
            visited.add(url)

            try:
                resp = requests.get(url, timeout=15)
                ctype = resp.headers.get("Content-Type", "")
                if "text/html" not in ctype:
                    continue
            except Exception as e:
                print(f"  !! failed {url}: {e}")
                continue

            soup = BeautifulSoup(resp.text, "html.parser")
            for tag in soup(["script", "style", "noscript"]):
                tag.decompose()

            text = soup.get_text(" ")
            if len(text.strip()) < 80:
                continue

            full_text = f"{url}\n\n{text}"
            chunks = chunk(full_text)
            meta = {
                "source_type": "website",
                "url": url,
                "domain": domain,
            }
            prefix = f"web-{url.replace('https://','').replace('http://','').replace('/','_')}"
            add_chunks(chunks, meta, prefix)

            # enqueue internal links
            for a in soup.find_all("a", href=True):
                href = a["href"].strip()
                if (
                    not href
                    or href.startswith("#")
                    or href.startswith("mailto:")
                    or href.startswith("tel:")
                ):
                    continue
                next_url = urljoin(url, href)
                parsed = urlparse(next_url)
                if parsed.netloc == domain and next_url not in visited:
                    queue.append(next_url)

    print(f"[web] done, crawled {len(visited)} pages")

# ---------- GitHub ----------

def is_text_file(path: str):
    ext = Path(path).suffix.lower()
    return ext in TEXT_EXT or ext in CODE_EXT

def should_skip(path: str):
    parts = path.split("/")
    return any(p in SKIP_DIRS for p in parts)

def ingest_github():
    if not (GITHUB_TOKEN and GITHUB_USERNAME and GITHUB_REPOS_ALLOWLIST):
        print("[github] missing config or allowlist, skipping")
        return

    print("[github] ingesting selected repos...")
    gh = Github(GITHUB_TOKEN)
    user = gh.get_user(GITHUB_USERNAME)
    all_repos = {r.name: r for r in user.get_repos()}

    for name in GITHUB_REPOS_ALLOWLIST:
        repo = all_repos.get(name)
        if not repo:
            print(f"  !! repo {name} not found under {GITHUB_USERNAME}")
            continue

        print(f"  [repo] {repo.full_name}")
        contents = repo.get_contents("")
        while contents:
            item = contents.pop(0)
            if item.type == "dir":
                if not should_skip(item.path):
                    contents.extend(repo.get_contents(item.path))
            else:
                if should_skip(item.path) or not is_text_file(item.path):
                    continue
                try:
                    content = item.decoded_content.decode("utf-8", errors="ignore")
                except Exception:
                    continue

                if len(content.strip()) < 80:
                    continue

                header = f"{repo.full_name} :: {item.path}\n"
                text = header + "\n" + content
                chunks = chunk(text)
                meta = {
                    "source_type": "github",
                    "repo": repo.full_name,
                    "path": item.path,
                    "url": f"https://github.com/{repo.full_name}/blob/HEAD/{item.path}",
                }
                prefix = f"gh-{repo.name}-{item.path.replace('/','_')}"
                add_chunks(chunks, meta, prefix)

    print("[github] done")

# ---------- public entry ----------

def build_index_if_empty():
    """
    Call this at startup: if collection is empty, ingest everything.
    """
    count = collection.count()
    if count > 0:
        print(f"[ingest] collection '{COLLECTION_NAME}' already has {count} records")
        return

    print(f"[ingest] building collection '{COLLECTION_NAME}' in {PERSIST_DIRECTORY}")
    os.makedirs(PERSIST_DIRECTORY, exist_ok=True)
    ingest_local_docs()
    ingest_websites()
    ingest_github()
    print("[ingest] complete")


if __name__ == "__main__":
    build_index_if_empty()
