import re
import string

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
    search_text = text.strip()
    if len(search_text) == 0:
        return []

    elements = re.split(r"(?<=[.!?])\s+", text.strip())
    if len(elements) == 1:
        punc_character = len([char for char in string.punctuation if elements[0].endswith(char)]) > 0
        elements[0] = f"{elements[0]}. " if not punc_character else elements[0]

    elements = list(
        filter(
            lambda x: len(x) > 0,
            list(map(lambda x: x.strip(), elements))
        )
    )

    start_index = 0
    chunks = []

    while start_index < len(elements):
        chunk_sentences = elements[start_index:start_index + size]
        if chunks and len(chunk_sentences) <= overlap:
            break
        chunks.append(" ".join(chunk_sentences))
        start_index += size - overlap

    return chunks