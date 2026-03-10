# /09 · Sistema de Q&A sobre Base de Conhecimento Interna

Projeto de teste técnico com prazo de 48h. O objetivo é construir um sistema que permita fazer perguntas em linguagem natural sobre documentos internos de uma empresa (500+ arquivos), retornando respostas com citação de fonte e score de confiança.

---

## Contexto

A empresa possui documentos internos em múltiplos formatos (PDF, DOCX, TXT). Hoje não existe forma de consultar esse conhecimento de forma centralizada — o sistema resolve isso via RAG (Retrieval-Augmented Generation): busca o trecho mais relevante nos documentos e usa um LLM para formular a resposta.

---

## Métricas do Projeto

| Custo             | Fonte                | Score                  | Prazo |
| ----------------- | -------------------- | ---------------------- | ----- |
| < €0,10 por query | Citação de parágrafo | Confiança por resposta | 48h   |

---

## O que construir

- Ingestão e chunking de documentos
- Embeddings com retrieval semântico
- Resposta com citação de parágrafo
- Score de confiança por resposta

## O que vai ser avaliado neste projeto

- Resposta correta com fonte certa?
- Funciona com documentos contraditórios?
- Custo por query com breakdown
- Falha graciosamente quando não sabe?

---

## Stack

| Camada       | Tecnologia                                 | Motivo                 |
| ------------ | ------------------------------------------ | ---------------------- |
| Embeddings   | `sentence-transformers` (all-MiniLM-L6-v2) | Grátis, roda local     |
| Vector Store | ChromaDB                                   | Grátis, simples, local |
| LLM          | Groq (free tier) — `llama-3.1-8b-instant`  | Grátis, rápido         |
| API          | FastAPI                                    | Leve e direto          |

---

## Estado Atual

- [x] Ambiente configurado (Python + venv + dependências)
- [x] Ingestão de documentos (PDF, DOCX, TXT) — `src/ingestao.py`
- [x] Geração de embeddings com sentence-transformers
- [x] Persistência no ChromaDB (`./banco`)
- [x] Sistema de busca semântica — `src/busca.py`
- [x] Geração de resposta com citação de fonte — `src/resposta.py`
- [x] Score de confiança (baseado em distância L2, limiar no top-1 chunk)
- [x] API FastAPI — `src/api.py`
- [x] UI simples (servida direto pela API em `GET /`)

---

## Como rodar

```bash
# 1. Ingerir documentos (só precisa rodar uma vez)
python src/ingestao.py

# 2. Subir a API
$env:PYTHONPATH="src"; uvicorn src.api:app --reload
```

Acesse `http://localhost:8000` para a UI e `http://localhost:8000/docs` para a documentação automática da API.

---

## Endpoints

| Método | Rota         | Descrição                                           |
| ------ | ------------ | --------------------------------------------------- |
| GET    | `/`          | UI web com campo de pergunta                        |
| POST   | `/perguntar` | Recebe `{"pergunta": "..."}`, retorna resposta JSON |
| GET    | `/status`    | Healthcheck com contagem de chunks no banco         |

---

## Decisões técnicas

| Decisão               | Escolha                              | Motivo                                                    |
| --------------------- | ------------------------------------ | --------------------------------------------------------- |
| Critério de corte     | Score do melhor chunk (top-1)        | Média penaliza queries com chunks irrelevantes            |
| Limiares de confiança | Alta < 0.80 · Média < 1.10           | Calibrado para português com all-MiniLM-L6-v2             |
| Recusa sem LLM        | score_top1 < 0.45                    | Custo zero para perguntas fora do escopo                  |
| Temperature           | 0.2                                  | Fidelidade ao contexto, menos alucinação                  |
| Prompt restritivo     | Regras explícitas contra especulação | Evita que o LLM complete com informações não documentadas |

---

## Custo real medido

| Query                         | Tokens in | Tokens out | Custo EUR   |
| ----------------------------- | --------- | ---------- | ----------- |
| Cancelar consulta             | 2680      | 83         | €0.000141   |
| Clareamento (recusou via LLM) | 1889      | 23         | €0.000096   |
| Vantagens alinhadores         | 1819      | 57         | €0.000096   |
| Marcar consulta               | 2069      | ~60        | €0.000105   |
| **Meta do projeto**           | —         | —          | **< €0,10** |
