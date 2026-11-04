from pathlib import Path
import hashlib
import re

import chromadb
import pymupdf as fitz
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent
DOCUMENTS_DIR = BASE_DIR / "documents"
DATABASE_DIR = BASE_DIR / "database"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "indala_admission"
CHUNK_SIZE = 180
CHUNK_OVERLAP = 35

_model = None


def get_embedding_model():
    """Load the embedding model once and reuse it."""
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def extract_pdf_pages(pdf_path: Path):
    """Return (page_number, text) pairs. Page numbers are 1-based."""
    pages = []
    with fitz.open(pdf_path) as pdf:
        for page_index, page in enumerate(pdf):
            text = page.get_text("text").strip()
            if text:
                pages.append((page_index + 1, text))
    return pages


def split_text(text: str, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping word chunks."""
    words = re.sub(r"\s+", " ", text).split()
    if not words:
        return []

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = end - overlap
    return chunks


def get_collection():
    DATABASE_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(DATABASE_DIR))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )


def delete_document_chunks(filename: str):
    """Delete all indexed chunks belonging to one source PDF."""
    collection = get_collection()
    collection.delete(where={"source": Path(filename).name})
    return collection.count()


def clear_all_chunks():
    """Delete every indexed chunk from the admission collection."""
    collection = get_collection()
    records = collection.get(include=["metadatas"])
    ids = records.get("ids", [])
    if ids:
        collection.delete(ids=ids)
    return collection.count()


def ingest_documents(pdf_paths=None):
    """Extract, chunk, embed and store PDFs in persistent ChromaDB."""
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    if pdf_paths is None:
        pdf_paths = sorted(DOCUMENTS_DIR.glob("*.pdf"))

    if not pdf_paths:
        raise ValueError(
            f"No PDF files found. Add PDFs to: {DOCUMENTS_DIR}"
        )

    collection = get_collection()
    model = get_embedding_model()

    all_ids, all_texts, all_metadatas = [], [], []
    skipped_pages = []

    for pdf_path in pdf_paths:
        pdf_path = Path(pdf_path)
        if pdf_path.suffix.lower() != ".pdf":
            continue

        pages = extract_pdf_pages(pdf_path)
        if not pages:
            skipped_pages.append(pdf_path.name)
            continue

        for page_number, page_text in pages:
            chunks = split_text(page_text)
            for chunk_index, chunk in enumerate(chunks):
                unique_text = f"{pdf_path.name}|{page_number}|{chunk_index}|{chunk}"
                chunk_id = hashlib.sha256(unique_text.encode("utf-8")).hexdigest()

                all_ids.append(chunk_id)
                all_texts.append(chunk)
                all_metadatas.append({
                    "source": pdf_path.name,
                    "page": int(page_number),
                    "chunk": int(chunk_index)
                })

    if not all_texts:
        raise ValueError(
            "No readable text was extracted. The PDFs may be scanned images; OCR is needed."
        )

    # Upsert makes re-processing identical documents safe.
    # Remove previous chunks for each successfully extracted source first,
    # so changed PDFs do not leave obsolete chunks behind.
    successfully_read_sources = sorted({
        metadata["source"] for metadata in all_metadatas
    })
    for source_name in successfully_read_sources:
        collection.delete(where={"source": source_name})

    batch_size = 64
    for start in range(0, len(all_texts), batch_size):
        end = start + batch_size
        batch_texts = all_texts[start:end]
        embeddings = model.encode(
            batch_texts,
            convert_to_numpy=True,
            normalize_embeddings=True
        ).tolist()

        collection.upsert(
            ids=all_ids[start:end],
            documents=batch_texts,
            metadatas=all_metadatas[start:end],
            embeddings=embeddings
        )

    return {
        "processed_files": len(successfully_read_sources),
        "stored_chunks": len(all_texts),
        "collection_count": collection.count(),
        "skipped_files": skipped_pages
    }


def retrieve_context(question: str, top_k=5):
    """Retrieve relevant chunks and return them with source metadata."""
    collection = get_collection()
    if collection.count() == 0:
        return []

    model = get_embedding_model()
    query_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).tolist()

    result = collection.query(
        query_embeddings=query_embedding,
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"]
    )

    matches = []
    for doc, metadata, distance in zip(
        result["documents"][0],
        result["metadatas"][0],
        result["distances"][0]
    ):
        matches.append({
            "text": doc,
            "source": metadata.get("source", "Unknown document"),
            "page": metadata.get("page", "Unknown"),
            "distance": float(distance)
        })
    return matches
