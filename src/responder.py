from config import MIN_RESPONSE_SCORE
from search import search
from llm import complete, calculate_cost

_ZERO_COST = calculate_cost(0, 0)

# Phrase the LLM uses when it can't find the answer in the excerpts
_LLM_REFUSAL_MARKER = "I could not find this information"


def _refuse(answer: str, chunks: list[dict], average_score: float) -> dict:
    return {
        "answer":        answer,
        "sources":       sorted({c["file"] for c in chunks}),
        "chunks":        [{"file": c["file"], "excerpt": c["excerpt"], "score": c["score"], "confidence": c["confidence"]} for c in chunks],
        "confidence":    "insufficient",
        "average_score": average_score,
        "cost":          _ZERO_COST,
        "refused":       True,
    }


def respond(question: str, top_k: int = 3) -> dict:
    chunks = search(question, top_k=top_k)

    if not chunks:
        print("[RESPONDER] No chunks returned — refusing.")
        return _refuse("No relevant documents found to answer the question.", [], 0.0)

    average_score = round(sum(c["score"] for c in chunks) / len(chunks), 4)
    best_score = chunks[0]["score"]
    best_distance = chunks[0]["distance"]

    print(f"[RESPONDER] best_distance={best_distance} best_score={best_score} average_score={average_score}")
    print(f"[RESPONDER] MIN_RESPONSE_SCORE={MIN_RESPONSE_SCORE}")
    print(f"[RESPONDER] Passes threshold? {best_score >= MIN_RESPONSE_SCORE}")

    if best_score < MIN_RESPONSE_SCORE:
        print("[RESPONDER] Score below threshold — refusing.")
        return _refuse(
            "Not enough information found in the documents to answer confidently.",
            chunks,
            average_score,
        )

    text, cost = complete(question, chunks)

    llm_refused = _LLM_REFUSAL_MARKER.lower() in text.lower()
    if llm_refused:
        print("[RESPONDER] LLM returned refusal phrase — downgrading confidence.")
        return {
            "answer":        text,
            "sources":       sorted({c["file"] for c in chunks}),
            "chunks":        [{"file": c["file"], "excerpt": c["excerpt"], "score": c["score"], "confidence": c["confidence"]} for c in chunks],
            "confidence":    "insufficient",
            "average_score": average_score,
            "cost":          cost,
            "refused":       True,
        }

    return {
        "answer":        text,
        "sources":       sorted({c["file"] for c in chunks}),
        "chunks":        [{"file": c["file"], "excerpt": c["excerpt"], "score": c["score"], "confidence": c["confidence"]} for c in chunks],
        "confidence":    chunks[0]["confidence"],
        "average_score": average_score,
        "cost":          cost,
        "refused":       False,
    }