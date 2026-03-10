import chromadb
from sentence_transformers import SentenceTransformer
from config import CHROMA_PATH, COLLECTION_NAME, EMBEDDING_MODEL

_model: SentenceTransformer | None = None
_client: chromadb.PersistentClient | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"Loading embedding model: {EMBEDDING_MODEL}")
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def get_collection():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=CHROMA_PATH)
    return _client.get_or_create_collection(
        COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def generate_embedding(text: str) -> list[float]:
    return get_model().encode(text).tolist()


def generate_embeddings(texts: list[str]) -> list[list[float]]:
    return get_model().encode(texts).tolist()


def chunk_exists(chunk_id: str) -> bool:
    return bool(get_collection().get(ids=[chunk_id])["ids"])


def add_chunk(chunk_id: str, embedding: list[float], text: str, metadata: dict) -> None:
    get_collection().add(
        ids=[chunk_id],
        embeddings=[embedding],
        documents=[text],
        metadatas=[metadata],
    )


def search_chunks(embedding: list[float], n_results: int) -> dict:
    return get_collection().query(
        query_embeddings=[embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )