import os
from groq import Groq
from dotenv import load_dotenv
from config import GROQ_MODEL, INPUT_COST_PER_TOKEN, OUTPUT_COST_PER_TOKEN

load_dotenv()

_client: Groq | None = None

SYSTEM_PROMPT = """You are an assistant that answers questions about internal documents of an orthodontic clinic.

Mandatory rules:
1. Answer ONLY the question asked. Do not add extra information even if it appears in the excerpts.
2. If the exact answer is not in the excerpts, respond ONLY with: "I could not find this information in the documents."
   Do not add anything after that. No context, no related information.
3. NEVER mention "EXCERPT 1", "EXCERPT 2" or any internal references.
4. Cite the source only after answering the question: (Source: file.docx)
5. No introductions like "Based on the documents..." or "According to the excerpts..."."""


def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return _client


def build_context(chunks: list[dict]) -> str:
    excerpts = [
        f"[EXCERPT {i} — Source: {chunk['file']}]\n{chunk['excerpt']}"
        for i, chunk in enumerate(chunks, 1)
    ]
    return "\n\n".join(excerpts)


def calculate_cost(tokens_input: int, tokens_output: int) -> dict:
    input_cost = tokens_input * INPUT_COST_PER_TOKEN
    output_cost = tokens_output * OUTPUT_COST_PER_TOKEN
    return {
        "tokens_input":    tokens_input,
        "tokens_output":   tokens_output,
        "input_cost_eur":  round(input_cost, 6),
        "output_cost_eur": round(output_cost, 6),
        "total_cost_eur":  round(input_cost + output_cost, 6),
    }


def complete(question: str, chunks: list[dict]) -> tuple[str, dict]:
    context = build_context(chunks)
    user_prompt = f"Document excerpts:\n\n{context}\n\n---\n\nQuestion: {question}"

    response = _get_client().chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=512,
    )

    text = response.choices[0].message.content.strip()
    cost = calculate_cost(response.usage.prompt_tokens, response.usage.completion_tokens)
    return text, cost