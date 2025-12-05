import re

def basic_chunking(text: str, size: int, overlap: int) -> list[str]:
    elements = text.split()
    start_index = 0
    chunks = []

    while start_index < len(elements):
        chunk = " ".join(elements[start_index:start_index + size])
        chunks.append(chunk)
        start_index += size - overlap

    return chunks

def semantic_chunking(text: str, size: int, overlap: int) -> list[str]:
    elements = re.split(r"(?<=[.!?])\s+", text)
    start_index = 0
    chunks = []

    while start_index < len(elements):
        chunk = " ".join(elements[start_index:start_index + size])
        chunks.append(chunk)
        start_index += size - overlap

    return chunks