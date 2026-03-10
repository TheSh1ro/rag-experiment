import os

_BASE = os.path.dirname(os.path.abspath(__file__))

CHROMA_PATH      = os.path.join(_BASE, "..", "database")
COLLECTION_NAME  = "documents"
EMBEDDING_MODEL  = "paraphrase-multilingual-MiniLM-L12-v2"
GROQ_MODEL       = "llama-3.1-8b-instant"

CHUNK_SIZE       = 500
CHUNK_OVERLAP    = 100
TOP_K            = 5

HIGH_CONFIDENCE_THRESHOLD   = 0.55
MEDIUM_CONFIDENCE_THRESHOLD = 0.85
MIN_RESPONSE_SCORE          = 0.58

INPUT_COST_PER_TOKEN  = 0.05 / 1_000_000
OUTPUT_COST_PER_TOKEN = 0.08 / 1_000_000