import os

from document_processor import read_document, split_into_chunks
from vector_store import generate_embeddings, chunk_exists, add_chunk


def ingest_file(path: str, file_name: str) -> None:
    text = read_document(path)
    if not text.strip():
        print(f"No text extracted: {file_name}")
        return

    chunks = split_into_chunks(text)
    embeddings = generate_embeddings(chunks)
    new_count = 0

    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        chunk_id = f"{file_name}_chunk_{i}"
        if not chunk_exists(chunk_id):
            add_chunk(chunk_id, embedding, chunk, {"arquivo": file_name, "chunk": i})
            new_count += 1

    print(f"{file_name}: {len(chunks)} chunks ({new_count} new added)")


def ingest_documents(folder: str = None) -> None:
    if folder is None:
        folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "documents")

    files = os.listdir(folder)
    print(f"{len(files)} file(s) found in '{folder}'")
    for file in files:
        ingest_file(os.path.join(folder, file), file)
    print("Ingestion complete.")


if __name__ == "__main__":
    ingest_documents()