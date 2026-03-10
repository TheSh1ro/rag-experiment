"""
busca.py — Busca semântica no ChromaDB

Fluxo:
  pergunta (texto)
    → embedding (mesmo modelo da ingestão)
      → busca vetorial no ChromaDB
        → chunks relevantes + score de confiança
"""

import chromadb
from sentence_transformers import SentenceTransformer

# ── Configurações ─────────────────────────────────────────────────────────────
# Mantém os mesmos valores do ingestao.py

CHROMA_PATH = "./banco"
NOME_COLECAO = "documentos"
MODELO_EMBEDDING = "all-MiniLM-L6-v2"
TOP_K = 5  # quantos chunks retornar por pergunta

# ChromaDB retorna distância L2: quanto MENOR a distância, mais similar o chunk
# Esses limiares foram calibrados para o modelo all-MiniLM-L6-v2
# Ajuste conforme seus testes com documentos reais
LIMIAR_ALTA   = 0.80   # distância < 0.80  → confiança ALTA
LIMIAR_MEDIA  = 1.10   # distância < 1.10  → confiança MÉDIA
#                        distância >= 1.10  → confiança BAIXA

# ── Inicialização (acontece uma vez quando o módulo é importado) ───────────────

print("Carregando modelo de embeddings...")
modelo = SentenceTransformer(MODELO_EMBEDDING)

chroma = chromadb.PersistentClient(path=CHROMA_PATH)
colecao = chroma.get_or_create_collection(NOME_COLECAO)

print("✅ Busca pronta.\n")


# ── Funções ───────────────────────────────────────────────────────────────────

def calcular_confianca(distancia: float) -> tuple[str, float]:
    """
    Converte a distância L2 do ChromaDB em label + percentual de confiança.

    A distância L2 vai de 0 (idêntico) a ~2 (completamente diferente).
    Convertemos para percentual invertendo: confianca = 1 - (distancia / 2)

    Retorna:
        label      → "alta", "média" ou "baixa"
        percentual → valor entre 0.0 e 1.0
    """
    percentual = max(0.0, 1.0 - (distancia / 2.0))

    if distancia < LIMIAR_ALTA:
        label = "alta"
    elif distancia < LIMIAR_MEDIA:
        label = "média"
    else:
        label = "baixa"

    return label, round(percentual, 4)


def buscar(pergunta: str, top_k: int = TOP_K) -> list[dict]:
    """
    Busca os chunks mais relevantes para uma pergunta.

    Args:
        pergunta: texto da pergunta em linguagem natural
        top_k:    número de resultados a retornar

    Retorna:
        Lista de dicts, cada um com:
            - trecho:      o texto do chunk encontrado
            - arquivo:     nome do arquivo de origem
            - chunk_index: índice do chunk dentro do arquivo
            - distancia:   distância L2 bruta (para debug)
            - confianca:   "alta", "média" ou "baixa"
            - score:       float entre 0 e 1 (1 = perfeito)
    """
    # 1. Transforma a pergunta em embedding (mesmo processo da ingestão)
    embedding_pergunta = modelo.encode(pergunta).tolist()

    # 2. Busca no ChromaDB
    resultados = colecao.query(
        query_embeddings=[embedding_pergunta],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    # 3. Monta a lista de resultados formatados
    chunks_encontrados = []

    documentos  = resultados["documents"][0]   # lista de textos
    metadados   = resultados["metadatas"][0]   # lista de dicts com arquivo/chunk
    distancias  = resultados["distances"][0]   # lista de floats

    for texto, meta, distancia in zip(documentos, metadados, distancias):
        label_confianca, score = calcular_confianca(distancia)

        chunks_encontrados.append({
            "trecho":      texto,
            "arquivo":     meta.get("arquivo", "desconhecido"),
            "chunk_index": meta.get("chunk", -1),
            "distancia":   round(distancia, 4),
            "confianca":   label_confianca,
            "score":       score,
        })

    return chunks_encontrados


def buscar_e_imprimir(pergunta: str, top_k: int = TOP_K) -> None:
    """
    Wrapper para testar a busca direto no terminal de forma legível.
    """
    print(f"\n{'='*60}")
    print(f"PERGUNTA: {pergunta}")
    print(f"{'='*60}")

    resultados = buscar(pergunta, top_k)

    if not resultados:
        print("Nenhum resultado encontrado.")
        return

    for i, r in enumerate(resultados, 1):
        print(f"\n[Resultado {i}]")
        print(f"  Arquivo:   {r['arquivo']} (chunk {r['chunk_index']})")
        print(f"  Confiança: {r['confianca'].upper()} ({r['score']*100:.1f}%)")
        print(f"  Distância: {r['distancia']} (debug)")
        print(f"  Trecho:    {r['trecho'][:300]}{'...' if len(r['trecho']) > 300 else ''}")

    print(f"\n{'='*60}\n")


# ── Teste direto ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Troque as perguntas abaixo por algo relacionado ao conteúdo dos seus docs
    buscar_e_imprimir("Posso desmarcar uma consulta que já agendei?")
    buscar_e_imprimir("Quanto custa fazer clareamento nos dentes?")
    buscar_e_imprimir("Quais são as vantagens dos alinhadores transparentes?")