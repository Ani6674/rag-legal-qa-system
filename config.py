"""Central configuration. Override LLM settings with environment variables."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
TESTSET_PATH = ROOT / "data" / "testset.json"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ---- Embedding models (short name -> HuggingFace id) ----
EMBEDDERS = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",   # max 256 word-pieces
    "bge-small": "BAAI/bge-small-en-v1.5",                # max 512 word-pieces
}
RERANKER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ---- Defaults ----
DEFAULT_CHUNK_SIZE = 200   # in WORDS (approx. 1.3 tokens per word)
DEFAULT_OVERLAP = 40
DEFAULT_K = 5

# ---- LLM backends: ollama | anthropic | gemini | hf ----
LLM_BACKEND = os.getenv("LLM_BACKEND", "ollama")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-haiku-5-5")  # needs ANTHROPIC_API_KEY
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")        # needs GEMINI_API_KEY; check current name
HF_MODEL = os.getenv("HF_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")      # runs locally / on Colab GPU

# ---- Map each PDF file name to its Act (used by make_testset.py) ----
# (full name, [lower-case aliases that must appear in a generated question])
ACT_NAMES = {
    "A2019-35.pdf": ("Consumer Protection Act, 2019", ["consumer protection act"]),
    "A2000-21.pdf": ("Information Technology Act, 2000", ["information technology act", "it act"]),
    "A2005-22.pdf": ("Right to Information Act, 2005", ["right to information act", "rti act"]),
}
