# Raven RAG Bot

RAG chatbot for Raven Mott's work:
- Crawls personal websites
- Indexes selected GitHub repos
- Indexes local PDFs/docs
- Serves a `/chat` endpoint for rmot1202.github.io

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
