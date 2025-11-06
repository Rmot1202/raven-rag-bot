import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env if present (useful locally)
load_dotenv()

# === Core paths ===
BASE_DIR = Path(__file__).resolve().parent.parent
PERSIST_DIRECTORY = os.getenv("PERSIST_DIRECTORY", str(BASE_DIR / "chroma_data"))
DOCS_DIR = Path(os.getenv("DOCS_DIR", BASE_DIR / "data" / "docs"))

# === OpenAI ===
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-5")

# === GitHub ===
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")  # create a fine-grained, read-only token
GITHUB_USERNAME = os.getenv("GITHUB_USERNAME", "Rmot1202")

# Only repos that represent your work (EDIT THIS LIST)
GITHUB_REPOS_ALLOWLIST = [
    "raven-rag-bot","Rmot1202.github.io","Cropnet","Rmot1202","Food-Website","malware-triage-pipeline","Gestura","BitFit_Part_2","Campgrounds.pt1","bit_2","bit","Campground-Pt3","Campgrounds","flixster_pt2","Flixster","and102-lab3-starter","Wishlist","Gmail","AdamJeddy","Wordle","SimpleCounter","meharry-summer-colab-intro-to-github","csm","MSCC25","Distributed-Malware-Analysis-System","Charity-Website","Decision-Tree-Classifier-Project","javascript-essential-training-2832077","Support-Vector-Machine-SVM-for-Neutron-Interaction-Classification","K-means-Clustering-Project","Heat-Raster-Analysis-for-Petersburg-VA","SNA-VSU-FACULTY","generative-ai-for-python-developers-4504249",
    
    
    # add/rename to match your actual repos
]

# === Websites to crawl (EDIT/ADD as needed) ===
SEED_SITES = [
    "https://rmot1202.github.io/",
    "www.linkedin.com/in/raven1202",
    "https://sresfs.net/",
    "https://ms-cc.org/",

]

MAX_WEB_PAGES = int(os.getenv("MAX_WEB_PAGES", "40"))

# === Chroma collection ===
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "raven_all")
