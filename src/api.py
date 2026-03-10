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
    return """
<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Q&A — Base de Conhecimento</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f4f5f7; color: #1a1a2e;
      min-height: 100vh; display: flex;
      align-items: center; justify-content: center; padding: 2rem;
    }
    .container {
      background: #fff; border-radius: 16px;
      box-shadow: 0 4px 24px rgba(0,0,0,0.08);
      padding: 2.5rem; width: 100%; max-width: 720px;
    }
    h1 { font-size: 1.4rem; font-weight: 700; margin-bottom: 0.3rem; }
    .subtitle { font-size: 0.875rem; color: #6b7280; margin-bottom: 2rem; }
    .input-row { display: flex; gap: 0.75rem; margin-bottom: 1.5rem; }
    input[type="text"] {
      flex: 1; padding: 0.75rem 1rem;
      border: 1.5px solid #e5e7eb; border-radius: 10px;
      font-size: 0.95rem; outline: none; transition: border-color 0.2s;
    }
    input[type="text"]:focus { border-color: #6366f1; }
    button {
      padding: 0.75rem 1.5rem; background: #6366f1; color: #fff;
      border: none; border-radius: 10px; font-size: 0.95rem;
      font-weight: 600; cursor: pointer; transition: background 0.2s; white-space: nowrap;
    }
    button:hover { background: #4f46e5; }
    button:disabled { background: #a5b4fc; cursor: not-allowed; }
    #result { display: none; }
    .card { border-radius: 12px; padding: 1.25rem 1.5rem; margin-bottom: 1rem; }
    .card-answer { background: #f0fdf4; border: 1.5px solid #bbf7d0; }
    .card-answer.refused { background: #fff7ed; border-color: #fed7aa; }
    .card-label {
      font-size: 0.75rem; font-weight: 700;
      text-transform: uppercase; letter-spacing: 0.06em;
      color: #6b7280; margin-bottom: 0.5rem;
    }
    .card-text { font-size: 0.95rem; line-height: 1.6; }
    .meta-row { display: flex; gap: 0.75rem; flex-wrap: wrap; }
    .badge {
      display: inline-flex; align-items: center; gap: 0.3rem;
      padding: 0.35rem 0.75rem; border-radius: 999px;
      font-size: 0.8rem; font-weight: 600;
    }
    .badge-high   { background: #dcfce7; color: #166534; }
    .badge-medium { background: #fef9c3; color: #854d0e; }
    .badge-low    { background: #fee2e2; color: #991b1b; }
    .badge-insuf  { background: #f3f4f6; color: #6b7280; }
    .badge-source { background: #ede9fe; color: #5b21b6; }
    .badge-cost   { background: #e0f2fe; color: #075985; }
    .loader { display: none; font-size: 0.875rem; color: #6b7280; margin-bottom: 1rem; }
    .loader.active { display: block; }
  </style>
</head>
<body>
<div class="container">
  <h1>Q&A — Knowledge Base</h1>
  <p class="subtitle">Ask a question in natural language about the internal documents.</p>
  <div class="input-row">
    <input type="text" id="question"
      placeholder="Ex: Can I cancel an appointment with less than 24h notice?"
      onkeydown="if(event.key==='Enter') askQuestion()" />
    <button id="btn" onclick="askQuestion()">Ask</button>
  </div>
  <p class="loader" id="loader">Searching documents...</p>
  <div id="result">
    <div class="card card-answer" id="card-answer">
      <div class="card-label">Answer</div>
      <div class="card-text" id="answer-text"></div>
    </div>
    <div class="meta-row" id="meta-row"></div>
  </div>
</div>
<script>
  const CONFIDENCE_CLASSES = {
    high: "badge-high", medium: "badge-medium",
    low: "badge-low", insufficient: "badge-insuf"
  };

  async function askQuestion() {
    const question = document.getElementById("question").value.trim();
    if (!question) return;

    const btn        = document.getElementById("btn");
    const loader     = document.getElementById("loader");
    const result     = document.getElementById("result");
    const cardAnswer = document.getElementById("card-answer");
    const answerText = document.getElementById("answer-text");
    const metaRow    = document.getElementById("meta-row");

    btn.disabled = true;
    loader.classList.add("active");
    result.style.display = "none";
    metaRow.innerHTML = "";

    try {
      const res  = await fetch("/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });
      const data = await res.json();

      answerText.textContent = data.answer;
      cardAnswer.className = "card card-answer" + (data.refused ? " refused" : "");

      const badgeConfidence = CONFIDENCE_CLASSES[data.confidence] || "badge-insuf";
      const scoreLabel      = (data.average_score * 100).toFixed(1);
      const badgesSources   = data.sources.map(s => `<span class="badge badge-source">${s}</span>`).join("");
      const badgeCost       = !data.refused
        ? `<span class="badge badge-cost">&euro;${data.cost.total_cost_eur.toFixed(6)} &middot; ${data.cost.tokens_input + data.cost.tokens_output} tokens</span>`
        : "";

      metaRow.innerHTML = `
        <span class="badge ${badgeConfidence}">Confidence: ${data.confidence.toUpperCase()} (${scoreLabel}%)</span>
        ${badgesSources}
        ${badgeCost}
      `;
      result.style.display = "block";
    } catch {
      answerText.textContent = "Error connecting to the API.";
      cardAnswer.className = "card card-answer refused";
      result.style.display = "block";
    } finally {
      btn.disabled = false;
      loader.classList.remove("active");
    }
  }
</script>
</body>
</html>
"""


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)