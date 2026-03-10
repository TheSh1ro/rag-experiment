from pypdf import PdfReader
from docx import Document
from config import CHUNK_SIZE, CHUNK_OVERLAP


def read_pdf(path: str) -> str:
    reader = PdfReader(path)
    return "\n".join(page.extract_text() for page in reader.pages)


def read_docx(path: str) -> str:
    doc = Document(path)
    return "\n".join(p.text for p in doc.paragraphs)


def read_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


_READERS = {"pdf": read_pdf, "docx": read_docx, "txt": read_txt}


def read_document(path: str) -> str:
    extension = path.lower().rsplit(".", 1)[-1]
    reader = _READERS.get(extension)
    return reader(path) if reader else ""


def split_into_chunks(
    text: str,
    size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    words = text.split()
    step = size - overlap
    return [
        " ".join(words[i : i + size])
        for i in range(0, len(words), step)
        if words[i : i + size]
    ]