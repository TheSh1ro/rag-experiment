# RAG Q&A System

A **Retrieval-Augmented Generation (RAG)** system that answers questions about internal documents in natural language. The project comes pre-configured with a set of sample documents and a themed web interface simulating a real company, so you can run it and see it working immediately.

---

## How it works

1. **Ingest** - Your `.pdf`, `.docx`, and `.txt` files are read, chunked, and stored as vector embeddings in a local ChromaDB database.
2. **Search** - When a question arrives, the system finds the most semantically similar document chunks using cosine similarity.
3. **Answer** - The top chunks are passed to a Groq-hosted LLM (Llama 3.1 8B), which generates a grounded answer with source citations.
4. **Confidence** - Every response includes a confidence score and label (`high`, `medium`, `low`, or `insufficient`), so you always know how reliable the answer is.

```
Question -> Embedding -> ChromaDB -> Top-K chunks -> Groq LLM -> Answer + Sources
```

---

## Project Structure

```
rag-qasystem/
├── documents/          <- Sample documents included (replace with your own if needed)
├── database/           <- ChromaDB persisted storage (auto-generated)
├── renderer/
│   └── index.html      <- Themed web UI (customize if needed)
└── src/
    ├── api.py               <- FastAPI endpoints (/ask, /status)
    ├── config.py            <- All tuneable settings in one place
    ├── document_processor.py  <- Reads and chunks documents
    ├── ingestion.py         <- Indexes documents into ChromaDB
    ├── vector_store.py      <- Embedding model + ChromaDB client
    ├── search.py            <- Semantic search + confidence scoring
    ├── llm.py               <- Groq API client + prompt + cost tracking
    └── responder.py         <- Business logic: thresholds + refusal handling
```

---

## Quick Start

### 1. Prerequisites

- Python 3.11+
- A [Groq API key](https://console.groq.com/) (free tier available)

### 2. Clone and install

```bash
git clone https://github.com/your-org/rag-qasystem.git
cd rag-qasystem

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
```

Open `.env` and fill in your key:

```env
GROQ_API_KEY=your_key_here
```

### 4. Ingest the sample documents

```bash
python src/ingestion.py
```

You will see output like:

```
4 file(s) found in 'documents'
Contexto Geral da Empresa.docx: 12 chunks (12 new added)
...
Ingestion complete.
```

### 5. Start the server

```bash
python src/api.py
```

- Web UI: http://localhost:8000
- Interactive API docs: http://localhost:8000/docs

The system is ready to answer questions about the included sample documents right away.

---

## Adapting for your own use

The project is intentionally pre-configured so you can evaluate it without any setup beyond the API key. When you are ready to use it with your own content, there are only two things to change:

**Documents** - Clear the `documents/` folder and add your own `.pdf`, `.docx`, or `.txt` files. Then delete the `database/` folder and run ingestion again so the vector store reflects the new content.

**Web interface** - Edit `renderer/index.html` to match your organization's branding and context. The file is self-contained and requires no build step.

Everything else (chunking, search, confidence scoring, cost tracking) works the same regardless of the documents you use.

---

## API Reference

### GET /status

Returns system health and stats.

```json
{
  "status": "ok",
  "chunks_in_db": 47,
  "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2",
  "llm_model": "llama-3.1-8b-instant"
}
```

### POST /ask

Ask a question in natural language.

**Request:**

```json
{ "question": "What are the payment methods accepted?" }
```

**Response:**

```json
{
  "answer": "The clinic accepts credit cards, bank transfer, and installment plans. (Source: Services.docx)",
  "sources": ["Services.docx"],
  "chunks": [
    {
      "file": "Services.docx",
      "excerpt": "...",
      "score": 0.87,
      "confidence": "high"
    }
  ],
  "confidence": "high",
  "average_score": 0.82,
  "cost": {
    "tokens_input": 312,
    "tokens_output": 48,
    "total_cost_eur": 0.00002
  },
  "refused": false
}
```

If the system cannot find a confident answer, `refused` will be `true` and no LLM cost is incurred.

---

## Configuration

All settings live in `src/config.py`:

| Parameter                     | Default                                 | Description                                        |
| ----------------------------- | --------------------------------------- | -------------------------------------------------- |
| `EMBEDDING_MODEL`             | `paraphrase-multilingual-MiniLM-L12-v2` | Local sentence-transformer for embeddings          |
| `GROQ_MODEL`                  | `llama-3.1-8b-instant`                  | LLM used for answer generation                     |
| `CHUNK_SIZE`                  | `500`                                   | Words per document chunk                           |
| `CHUNK_OVERLAP`               | `100`                                   | Overlapping words between consecutive chunks       |
| `TOP_K`                       | `5`                                     | Number of chunks retrieved per query               |
| `HIGH_CONFIDENCE_THRESHOLD`   | `0.55`                                  | Cosine distance below which confidence is `high`   |
| `MEDIUM_CONFIDENCE_THRESHOLD` | `0.85`                                  | Cosine distance below which confidence is `medium` |
| `MIN_RESPONSE_SCORE`          | `0.58`                                  | Minimum score to attempt an LLM call               |

---

## Running Tests

```bash
pytest tests/test_rag.py -v
```

The test suite covers:

- **Document processor** - chunking logic, overlap correctness, edge cases
- **Search** - confidence label thresholds, score range validation
- **LLM** - context building, cost calculation
- **Responder** - refusal rules, score thresholds, source deduplication
- **API** - HTTP endpoints, error handling, response schema validation

All external dependencies (Groq API, ChromaDB) are mocked, so tests run fully offline.

---

## Design Decisions

**Why local embeddings?** The `paraphrase-multilingual-MiniLM-L12-v2` model runs entirely on your machine. No data leaves your infrastructure during ingestion or search. Only the final answer generation call reaches Groq's API.

**Why refuse instead of guess?** The system has a hard score threshold (`MIN_RESPONSE_SCORE`). If retrieved chunks are not confident enough, it returns a clear `refused: true` response rather than sending weak context to the LLM. This prevents hallucinated answers.

**Cost transparency** - Every response includes exact token counts and cost in EUR, so you can monitor spend without surprises.

---

## Tech Stack

| Layer            | Technology                      |
| ---------------- | ------------------------------- |
| API              | FastAPI + Uvicorn               |
| Vector DB        | ChromaDB (local, persistent)    |
| Embeddings       | sentence-transformers (offline) |
| LLM              | Groq - Llama 3.1 8B Instant     |
| Document parsing | pypdf, python-docx              |
| Tests            | pytest + unittest.mock          |

---

## License

MIT
