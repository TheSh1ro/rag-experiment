from pathlib import Path
import sys, os

sys.path.insert(0, os.path.dirname(__file__))

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from responder import respond
from vector_store import get_collection
from config import EMBEDDING_MODEL, GROQ_MODEL

app = FastAPI(
    title="Q&A Knowledge Base",
    description="RAG system for querying internal documents",
    version="1.0.0",
)


class QuestionRequest(BaseModel):
    question: str


class CostResponse(BaseModel):
    tokens_input: int
    tokens_output: int
    total_cost_eur: float


class AnswerResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence: str
    average_score: float
    cost: CostResponse
    refused: bool


@app.get("/status")
def status():
    return {
        "status": "ok",
        "chunks_in_db": get_collection().count(),
        "embedding_model": EMBEDDING_MODEL,
        "llm_model": GROQ_MODEL,
    }


@app.post("/ask", response_model=AnswerResponse)
def ask(body: QuestionRequest):
    if len(body.question.strip()) < 5:
        return AnswerResponse(
            answer="Please ask a complete question.",
            sources=[],
            confidence="insufficient",
            average_score=0.0,
            cost=CostResponse(tokens_input=0, tokens_output=0, total_cost_eur=0.0),
            refused=True,
        )
    try:
        result = respond(body.question)
    except Exception as exc:
        return AnswerResponse(
            answer=f"Internal error while processing the question: {exc}",
            sources=[],
            confidence="insufficient",
            average_score=0.0,
            cost=CostResponse(tokens_input=0, tokens_output=0, total_cost_eur=0.0),
            refused=True,
        )
    return AnswerResponse(
        answer=result["answer"],
        sources=result["sources"],
        confidence=result["confidence"],
        average_score=result["average_score"],
        cost=CostResponse(**{
            k: result["cost"][k]
            for k in ["tokens_input", "tokens_output", "total_cost_eur"]
        }),
        refused=result["refused"],
    )


@app.get("/", response_class=HTMLResponse)
def ui():
    return (Path(__file__).parent / "index.html").read_text(encoding="utf-8")


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)