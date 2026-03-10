"""
api.py — API FastAPI com UI embutida

Endpoints:
  GET  /          → página HTML com campo de pergunta
  POST /perguntar → recebe pergunta, retorna resposta + fonte + score + custo
  GET  /status    → healthcheck com contagem de chunks no banco

Rodar:
  uvicorn src.api:app --reload
  ou
  python src/api.py
"""
import sys, os

# ── Path (deve vir antes dos imports locais) ───────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from resposta import responder
from busca import colecao

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Q&A Base de Conhecimento",
    description="Sistema RAG para consulta de documentos internos",
    version="1.0.0",
)

# ── Modelos de request/response ───────────────────────────────────────────────

class PerguntaRequest(BaseModel):
    pergunta: str

class CustoResponse(BaseModel):
    tokens_input: int
    tokens_output: int
    custo_total_eur: float

class RespostaResponse(BaseModel):
    resposta: str
    fontes: list[str]
    confianca: str
    score_medio: float
    custo: CustoResponse
    recusou: bool

# ── Endpoints JSON ────────────────────────────────────────────────────────────

@app.get("/status")
def status():
    """Healthcheck — confirma que a API está de pé e mostra chunks no banco."""
    total_chunks = colecao.count()
    return {
        "status": "ok",
        "chunks_no_banco": total_chunks,
        "modelo_embedding": "all-MiniLM-L6-v2",
        "modelo_llm": "llama-3.1-8b-instant",
    }


@app.post("/perguntar", response_model=RespostaResponse)
def perguntar(body: PerguntaRequest):
    """Recebe uma pergunta e retorna resposta com citação, score e custo."""
    try:
        resultado = responder(body.pergunta)
    except Exception as exc:
        return RespostaResponse(
            resposta=f"Erro interno ao processar a pergunta: {exc}",
            fontes=[],
            confianca="insuficiente",
            score_medio=0.0,
            custo=CustoResponse(tokens_input=0, tokens_output=0, custo_total_eur=0.0),
            recusou=True,
        )
    return RespostaResponse(
        resposta=resultado["resposta"],
        fontes=resultado["fontes"],
        confianca=resultado["confianca"],
        score_medio=resultado["score_medio"],
        custo=CustoResponse(**{
            k: resultado["custo"][k]
            for k in ["tokens_input", "tokens_output", "custo_total_eur"]
        }),
        recusou=resultado["recusou"],
    )


# ── UI HTML ───────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def ui():
    """Interface web simples servida direto pela API."""
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
      background: #f4f5f7;
      color: #1a1a2e;
      min-height: 100vh;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 2rem;
    }

    .container {
      background: #fff;
      border-radius: 16px;
      box-shadow: 0 4px 24px rgba(0,0,0,0.08);
      padding: 2.5rem;
      width: 100%;
      max-width: 720px;
    }

    h1 {
      font-size: 1.4rem;
      font-weight: 700;
      margin-bottom: 0.3rem;
      color: #1a1a2e;
    }

    .subtitle {
      font-size: 0.875rem;
      color: #6b7280;
      margin-bottom: 2rem;
    }

    .input-row {
      display: flex;
      gap: 0.75rem;
      margin-bottom: 1.5rem;
    }

    input[type="text"] {
      flex: 1;
      padding: 0.75rem 1rem;
      border: 1.5px solid #e5e7eb;
      border-radius: 10px;
      font-size: 0.95rem;
      outline: none;
      transition: border-color 0.2s;
    }

    input[type="text"]:focus { border-color: #6366f1; }

    button {
      padding: 0.75rem 1.5rem;
      background: #6366f1;
      color: #fff;
      border: none;
      border-radius: 10px;
      font-size: 0.95rem;
      font-weight: 600;
      cursor: pointer;
      transition: background 0.2s;
      white-space: nowrap;
    }

    button:hover { background: #4f46e5; }
    button:disabled { background: #a5b4fc; cursor: not-allowed; }

    /* Resultado */
    #resultado { display: none; }

    .card {
      border-radius: 12px;
      padding: 1.25rem 1.5rem;
      margin-bottom: 1rem;
    }

    .card-resposta {
      background: #f0fdf4;
      border: 1.5px solid #bbf7d0;
    }

    .card-resposta.recusou {
      background: #fff7ed;
      border-color: #fed7aa;
    }

    .card-label {
      font-size: 0.75rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: #6b7280;
      margin-bottom: 0.5rem;
    }

    .card-texto {
      font-size: 0.95rem;
      line-height: 1.6;
      color: #1a1a2e;
    }

    .meta-row {
      display: flex;
      gap: 0.75rem;
      flex-wrap: wrap;
    }

    .badge {
      display: inline-flex;
      align-items: center;
      gap: 0.3rem;
      padding: 0.35rem 0.75rem;
      border-radius: 999px;
      font-size: 0.8rem;
      font-weight: 600;
    }

    .badge-alta    { background: #dcfce7; color: #166534; }
    .badge-media   { background: #fef9c3; color: #854d0e; }
    .badge-baixa   { background: #fee2e2; color: #991b1b; }
    .badge-insuf   { background: #f3f4f6; color: #6b7280; }
    .badge-fonte   { background: #ede9fe; color: #5b21b6; }
    .badge-custo   { background: #e0f2fe; color: #075985; }

    .loader {
      display: none;
      font-size: 0.875rem;
      color: #6b7280;
      margin-bottom: 1rem;
    }

    .loader.ativo { display: block; }
  </style>
</head>
<body>
<div class="container">
  <h1>📄 Q&A — Base de Conhecimento</h1>
  <p class="subtitle">Faça uma pergunta em linguagem natural sobre os documentos internos.</p>

  <div class="input-row">
    <input
      type="text"
      id="pergunta"
      placeholder="Ex: Posso cancelar uma consulta com menos de 24h?"
      onkeydown="if(event.key==='Enter') perguntar()"
    />
    <button id="btn" onclick="perguntar()">Perguntar</button>
  </div>

  <p class="loader" id="loader">⏳ Buscando nos documentos...</p>

  <div id="resultado">
    <div class="card card-resposta" id="card-resposta">
      <div class="card-label">Resposta</div>
      <div class="card-texto" id="texto-resposta"></div>
    </div>

    <div class="meta-row" id="meta-row"></div>
  </div>
</div>

<script>
  async function perguntar() {
    const input = document.getElementById("pergunta");
    const pergunta = input.value.trim();
    if (!pergunta) return;

    const btn = document.getElementById("btn");
    const loader = document.getElementById("loader");
    const resultado = document.getElementById("resultado");
    const cardResposta = document.getElementById("card-resposta");
    const textoResposta = document.getElementById("texto-resposta");
    const metaRow = document.getElementById("meta-row");

    btn.disabled = true;
    loader.classList.add("ativo");
    resultado.style.display = "none";
    metaRow.innerHTML = "";

    try {
      const res = await fetch("/perguntar", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ pergunta }),
      });

      const data = await res.json();

      // Resposta
      textoResposta.textContent = data.resposta;
      cardResposta.className = "card card-resposta" + (data.recusou ? " recusou" : "");

      // Badges de confiança
      const confiancaClasse = {
        alta: "badge-alta", média: "badge-media",
        baixa: "badge-baixa", insuficiente: "badge-insuf"
      }[data.confianca] || "badge-insuf";

      const scoreLabel = (data.score_medio * 100).toFixed(1);

      metaRow.innerHTML = `
        <span class="badge ${confiancaClasse}">
          Confiança: ${data.confianca.toUpperCase()} (${scoreLabel}%)
        </span>
        ${data.fontes.map(f => `<span class="badge badge-fonte">📄 ${f}</span>`).join("")}
        ${!data.recusou
          ? `<span class="badge badge-custo">
               €${data.custo.custo_total_eur.toFixed(6)}
               &nbsp;·&nbsp; ${data.custo.tokens_input + data.custo.tokens_output} tokens
             </span>`
          : ""}
      `;

      resultado.style.display = "block";

    } catch (err) {
      textoResposta.textContent = "Erro ao conectar com a API.";
      cardResposta.className = "card card-resposta recusou";
      resultado.style.display = "block";
    } finally {
      btn.disabled = false;
      loader.classList.remove("ativo");
    }
  }
</script>
</body>
</html>
"""

# ── Entrypoint direto ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)