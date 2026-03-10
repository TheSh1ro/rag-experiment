import os
import chromadb
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from docx import Document
from dotenv import load_dotenv

load_dotenv()

modelo_embedding = SentenceTransformer("all-MiniLM-L6-v2")
chroma = chromadb.PersistentClient(path="./banco")
colecao = chroma.get_or_create_collection("documentos")

def ler_pdf(caminho):
    reader = PdfReader(caminho)
    texto = ""
    for pagina in reader.pages:
        texto += pagina.extract_text() + "\n"
    return texto

def ler_docx(caminho):
    doc = Document(caminho)
    return "\n".join([p.text for p in doc.paragraphs])

def ler_txt(caminho):
    with open(caminho, "r", encoding="utf-8") as f:
        return f.read()

def ler_documento(caminho):
    ext = caminho.lower().split(".")[-1]
    if ext == "pdf":
        return ler_pdf(caminho)
    elif ext == "docx":
        return ler_docx(caminho)
    elif ext == "txt":
        return ler_txt(caminho)
    return ""

def dividir_em_chunks(texto, tamanho=500, overlap=100):
    """
    Divide o texto em chunks com sobreposição (overlap) entre janelas.
    O overlap evita que conceitos sejam cortados na fronteira entre chunks,
    melhorando o retrieval semântico.

    Args:
        tamanho:  número de palavras por chunk
        overlap:  número de palavras repetidas entre chunks consecutivos
    """
    palavras = texto.split()
    chunks = []
    passo = tamanho - overlap
    for i in range(0, len(palavras), passo):
        chunk = " ".join(palavras[i:i+tamanho])
        if chunk:
            chunks.append(chunk)
    return chunks

def ingerir_documentos(pasta="./documentos"):
    arquivos = os.listdir(pasta)
    print(f"Encontrei {len(arquivos)} arquivo(s)")

    for arquivo in arquivos:
        caminho = os.path.join(pasta, arquivo)
        print(f"Processando: {arquivo}")

        texto = ler_documento(caminho)
        if not texto.strip():
            print(f"  ⚠️ Sem texto: {arquivo}")
            continue

        chunks = dividir_em_chunks(texto)
        print(f"  → {len(chunks)} chunks gerados")

        embeddings = modelo_embedding.encode(chunks).tolist()

        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_id = f"{arquivo}_chunk_{i}"

            # Evita duplicatas: ignora chunks já existentes no banco
            existente = colecao.get(ids=[chunk_id])
            if existente["ids"]:
                continue

            colecao.add(
                ids=[chunk_id],
                embeddings=[embedding],
                documents=[chunk],
                metadatas=[{"arquivo": arquivo, "chunk": i}]
            )

        print(f"  ✅ {arquivo} salvo no banco")

if __name__ == "__main__":
    ingerir_documentos()
    print("\n✅ Ingestão concluída!")