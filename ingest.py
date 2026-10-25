"""Document loading, cleaning and chunking."""
import re
from pathlib import Path

from pypdf import PdfReader


def clean(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"-\n(?=[a-z])", "", text)      # re-join hyphenated line breaks
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_documents(raw_dir) -> list[dict]:
    """Return one record per PDF page / text file: {source, page, text}."""
    docs = []
    for p in sorted(Path(raw_dir).glob("*")):
        suffix = p.suffix.lower()
        if suffix == ".pdf":
            reader = PdfReader(str(p))
            for i, page in enumerate(reader.pages):
                text = clean(page.extract_text() or "")
                if text:
                    docs.append({"source": p.name, "page": i + 1, "text": text})
        elif suffix in (".txt", ".md"):
            docs.append({"source": p.name, "page": 1,
                         "text": clean(p.read_text(encoding="utf-8", errors="ignore"))})
    if not docs:
        raise FileNotFoundError(f"No PDF/TXT files found in {raw_dir}. See README for data sources.")
    return docs


def chunk_documents(docs: list[dict], chunk_size: int = 200, overlap: int = 40) -> list[dict]:
    """Sliding-window chunking by words, per source file (chunks may span pages).

    Each chunk keeps the page number where it starts, for citations.
    """
    assert 0 <= overlap < chunk_size, "overlap must be smaller than chunk_size"
    by_source: dict[str, list[dict]] = {}
    for d in docs:
        by_source.setdefault(d["source"], []).append(d)

    chunks = []
    step = chunk_size - overlap
    for source, pages in by_source.items():
        words, word_page = [], []
        for d in pages:
            w = d["text"].split()
            words += w
            word_page += [d["page"]] * len(w)
        for start in range(0, len(words), step):
            piece = words[start:start + chunk_size]
            if start > 0 and len(piece) < 20:      # drop tiny tail
                break
            chunks.append({
                "id": f"{source}::{len(chunks)}",
                "source": source,
                "page": word_page[start],
                "text": " ".join(piece),
            })
            if start + chunk_size >= len(words):
                break
    return chunks
