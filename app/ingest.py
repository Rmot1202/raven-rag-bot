import os, re, json
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from github import Github
from openai import OpenAI
from pypdf import PdfReader
import docx

from .config import (
    BASE_DIR,
    DOCS_DIR,
    OPENAI_API_KEY,
    OPENAI_EMBED_MODEL,
    GITHUB_TOKEN,
    GITHUB_USERNAME,
    GITHUB_REPOS_ALLOWLIST,
    SEED_SITES,
    MAX_WEB_PAGES,
)

if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY")

client = OpenAI()

INDEX_PATH = BASE_DIR / "data" / "index.json"

TEXT_EXT = {".md", ".txt"}
CODE_EXT = {".py", ".ipynb", ".kt", ".java", ".c", ".cpp", ".js", ".ts", ".rs", ".go", ".rb", ".php"}
SKIP_DIRS = {".git", "node_modules", "build", "dist", ".gradle", ".idea", "__pycache__"}


def clean_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def chunk(text: str, max_tokens=400, overlap=80):
    text = clean_ws(text)
    if not text:
        return []
    max_len = max_tokens * 4  # rough chars-per-token
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


def load_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def load_docx(path: Path) -> str:
    doc = docx.Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs)


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def ingest_local_docs(entries):
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
        if not chunks:
            continue

        embs = embed_batch(chunks)
        for c, e in zip(chunks, embs):
            entries.append({
                "embedding": e,
                "text": c,
                "meta": {
                    "source_type": "local_doc",
                    "filename": label,
                }
            })


def ingest_websites(entries):
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
                if "text/html" not in resp.headers.get("Content-Type", ""):
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

            full = f"{url}\n\n{text}"
            chunks = chunk(full)
            if not chunks:
                continue

            embs = embed_batch(chunks)
            for c, e in zip(chunks, embs):
                entries.append({
                    "embedding": e,
                    "text": c,
                    "meta": {
                        "source_type": "website",
                        "url": url,
                        "domain": domain,
                    }
                })

            for a in soup.find_all("a", href=True):
                href = a["href"].strip()
                if not href or href.startswith("#") or href.startswith("mailto:") or href.startswith("tel:"):
                    continue
                nxt = urljoin(url, href)
                parsed = urlparse(nxt)
                if parsed.netloc == domain and nxt not in visited:
                    queue.append(nxt)

    print(f"[web] done, crawled {len(visited)} pages")


def is_text_file(path: str):
    ext = Path(path).suffix.lower()
    return ext in TEXT_EXT or ext in CODE_EXT


def should_skip(path: str):
    parts = path.split("/")
    return any(p in SKIP_DIRS for p in parts)


def ingest_github(entries):
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
                if not chunks:
                    continue

                embs = embed_batch(chunks)
                for c, e in zip(chunks, embs):
                    entries.append({
                        "embedding": e,
                        "text": c,
                        "meta": {
                            "source_type": "github",
                            "repo": repo.full_name,
                            "path": item.path,
                            "url": f"https://github.com/{repo.full_name}/blob/HEAD/{item.path}",
                        }
                    })

    print("[github] done")


def build_index_if_missing():
    # if index exists and is non-empty, skip
    if INDEX_PATH.exists():
        try:
            with INDEX_PATH.open("r") as f:
                data = json.load(f)
            if isinstance(data, list) and data:
                print(f"[ingest] index already exists with {len(data)} entries")
                return
        except Exception:
            pass

    print("[ingest] building index...")
    entries = []
    os.makedirs(INDEX_PATH.parent, exist_ok=True)

    ingest_local_docs(entries)
    ingest_websites(entries)
    ingest_github(entries)

    with INDEX_PATH.open("w") as f:
        json.dump(entries, f)

    print(f"[ingest] saved {len(entries)} entries to {INDEX_PATH}")
