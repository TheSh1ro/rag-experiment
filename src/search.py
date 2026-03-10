from config import TOP_K, HIGH_CONFIDENCE_THRESHOLD, MEDIUM_CONFIDENCE_THRESHOLD
from vector_store import generate_embedding, search_chunks


def _calculate_confidence(distance: float) -> tuple[str, float]:
    percentage = max(0.0, 1.0 - (distance / 2.0))
    if distance < HIGH_CONFIDENCE_THRESHOLD:
        label = "high"
    elif distance < MEDIUM_CONFIDENCE_THRESHOLD:
        label = "medium"
    else:
        label = "low"
    return label, round(percentage, 4)


def search(question: str, top_k: int = TOP_K) -> list[dict]:
    print(f"\n[SEARCH] Question: {question!r}")
    embedding = generate_embedding(question)
    results = search_chunks(embedding, top_k)

    print(f"[SEARCH] Raw distances: {results['distances'][0]}")

    chunks = []
    for text, metadata, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        label, score = _calculate_confidence(distance)
        print(f"[SEARCH] file={metadata.get('arquivo')} distance={distance:.4f} confidence={label} score={score}")
        chunks.append({
            "excerpt":     text,
            "file":        metadata.get("arquivo", "unknown"),
            "chunk_index": metadata.get("chunk", -1),
            "distance":    round(distance, 4),
            "confidence":  label,
            "score":       score,
        })

    return chunks