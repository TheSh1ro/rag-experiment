"""
resposta.py — Geração de resposta com citação de fonte via Groq

Fluxo:
  pergunta
    → busca.py (chunks relevantes + scores)
      → decisão: responder ou recusar?
        → se responder: Groq gera resposta com citação
          → retorna resposta + fonte + score + custo estimado
"""

import os
from groq import Groq
from dotenv import load_dotenv
from busca import buscar

load_dotenv()

# ── Configurações ─────────────────────────────────────────────────────────────

MODELO_GROQ = "llama-3.1-8b-instant"

# Score mínimo do MELHOR chunk para tentar responder
# Abaixo disso: recusa educada, sem chamar o Groq (custo zero)
SCORE_MINIMO = 0.45

# Custo estimado do Groq (free tier = grátis, mas calculamos mesmo assim)
# llama3-8b: ~$0.05 por 1M tokens de input, ~$0.08 por 1M de output
CUSTO_INPUT_POR_TOKEN  = 0.05 / 1_000_000
CUSTO_OUTPUT_POR_TOKEN = 0.08 / 1_000_000

# ── Cliente Groq ──────────────────────────────────────────────────────────────

cliente = Groq(api_key=os.getenv("GROQ_API_KEY"))


# ── Funções ───────────────────────────────────────────────────────────────────

def montar_contexto(chunks: list[dict]) -> str:
    """
    Transforma a lista de chunks num bloco de texto numerado para o prompt.
    Cada trecho fica identificado com seu arquivo de origem.
    """
    linhas = []
    for i, chunk in enumerate(chunks, 1):
        linhas.append(
            f"[TRECHO {i} — Fonte: {chunk['arquivo']}]\n{chunk['trecho']}"
        )
    return "\n\n".join(linhas)


def calcular_custo(tokens_input: int, tokens_output: int) -> dict:
    custo_input  = tokens_input  * CUSTO_INPUT_POR_TOKEN
    custo_output = tokens_output * CUSTO_OUTPUT_POR_TOKEN
    total        = custo_input + custo_output
    return {
        "tokens_input":  tokens_input,
        "tokens_output": tokens_output,
        "custo_input_eur":  round(custo_input, 6),
        "custo_output_eur": round(custo_output, 6),
        "custo_total_eur":  round(total, 6),
    }


def responder(pergunta: str, top_k: int = 3) -> dict:
    """
    Função principal. Recebe a pergunta e retorna um dict com:
        - resposta:    texto gerado (ou mensagem de recusa)
        - fontes:      lista de arquivos citados
        - confianca:   "alta", "média", "baixa" ou "insuficiente"
        - score_medio: float médio dos chunks usados
        - custo:       breakdown de tokens e custo em EUR
        - recusou:     True se não havia contexto suficiente
    """

    # 1. Busca os chunks mais relevantes
    chunks = buscar(pergunta, top_k=top_k)

    if not chunks:
        return {
            "resposta":    "Não encontrei nenhum documento relevante para responder.",
            "fontes":      [],
            "confianca":   "insuficiente",
            "score_medio": 0.0,
            "custo":       calcular_custo(0, 0),
            "recusou":     True,
        }

    # 2. Calcula scores
    score_medio = round(sum(c["score"] for c in chunks) / len(chunks), 4)
    score_top1  = chunks[0]["score"]   # melhor chunk — critério de corte

    # 3. Decide se responde ou recusa (sem chamar o Groq)
    if score_top1 < SCORE_MINIMO:
        fontes_baixas = sorted({c["arquivo"] for c in chunks})
        return {
            "resposta": (
                "Não encontrei informações suficientes nos documentos disponíveis "
                "para responder com segurança a essa pergunta."
            ),
            "fontes":      fontes_baixas,
            "confianca":   "insuficiente",
            "score_medio": score_medio,
            "custo":       calcular_custo(0, 0),
            "recusou":     True,
        }

    # 4. Monta o contexto e o prompt
    contexto = montar_contexto(chunks)

    prompt_sistema = """Você é um assistente que responde perguntas sobre documentos internos de uma clínica ortodôntica.

Regras obrigatórias:
1. Responda APENAS a pergunta feita. Não adicione informações extras mesmo que estejam nos trechos.
2. Se a resposta exata não estiver nos trechos, responda SOMENTE: "Não encontrei essa informação nos documentos."
   Não adicione nada depois disso. Nem contexto, nem informações relacionadas.
3. NUNCA mencione "TRECHO 1", "TRECHO 2" ou qualquer referência interna.
4. Cite a fonte apenas quando tiver respondido a pergunta: (Fonte: arquivo.docx)
5. Sem introduções como "Com base nos documentos..." ou "De acordo com os trechos..."."""

    prompt_usuario = f"""Trechos dos documentos:

{contexto}

---

Pergunta: {pergunta}"""

    # 5. Chama o Groq
    resposta_groq = cliente.chat.completions.create(
        model=MODELO_GROQ,
        messages=[
            {"role": "system", "content": prompt_sistema},
            {"role": "user",   "content": prompt_usuario},
        ],
        temperature=0.2,   # baixo = mais fiel ao contexto, menos criativo
        max_tokens=512,
    )

    texto_resposta = resposta_groq.choices[0].message.content.strip()

    # 6. Extrai uso de tokens para calcular custo
    uso = resposta_groq.usage
    custo = calcular_custo(uso.prompt_tokens, uso.completion_tokens)

    # 7. Reutiliza o label de confiança já calculado em busca.py para o top-1 chunk,
    #    evitando divergência com os limiares definidos lá.
    #    busca.py: alta → distância < 0.80 (score > 0.60)
    #              média → distância < 1.10 (score > 0.45)
    confianca = chunks[0]["confianca"]

    fontes = sorted({c["arquivo"] for c in chunks})

    return {
        "resposta":    texto_resposta,
        "fontes":      fontes,
        "confianca":   confianca,
        "score_medio": score_medio,
        "custo":       custo,
        "recusou":     False,
    }


def responder_e_imprimir(pergunta: str) -> None:
    """Wrapper para testes no terminal."""
    print(f"\n{'='*60}")
    print(f"PERGUNTA: {pergunta}")
    print(f"{'='*60}")

    r = responder(pergunta)

    print(f"\nRESPOSTA:\n{r['resposta']}")
    print(f"\nFONTES:     {', '.join(r['fontes']) if r['fontes'] else '—'}")
    print(f"CONFIANÇA:  {r['confianca'].upper()}  (score médio: {r['score_medio']})")

    if not r["recusou"]:
        c = r["custo"]
        print(f"CUSTO:      €{c['custo_total_eur']:.6f}  "
              f"({c['tokens_input']} tokens in / {c['tokens_output']} tokens out)")

    print(f"\n{'='*60}\n")


# ── Teste direto ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    responder_e_imprimir("Posso desmarcar uma consulta que já agendei?")
    responder_e_imprimir("Quanto custa fazer clareamento nos dentes?")
    responder_e_imprimir("Quais são as vantagens dos alinhadores transparentes?")