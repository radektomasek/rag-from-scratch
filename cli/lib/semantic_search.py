import os
import numpy as np
from sentence_transformers import SentenceTransformer

from .utils import data_read

cache_dir = './cache'
db_file = 'movie_embeddings.npy'

class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def generate_embedding(self, text: str):
        if text.strip() == "" or text.isspace():
            raise ValueError("Parameter text is empty")

        result = self.model.encode([text])
        return result[0]

    def build_embeddings(self, documents: list[dict]):
        movies = []
        self.documents = documents

        for document in documents:
            doc_id = document.get("id")
            if not doc_id:
                raise ValueError("Missing 'id' in the document element")
            self.document_map[doc_id] = document

            data = f"{document['title']}: {document['description']}"
            movies.append(data)

        self.embeddings = self.model.encode(movies, show_progress_bar=True)
        np.save(os.path.join(cache_dir, db_file), self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents: list[dict]):
        if os.path.exists(os.path.join(cache_dir, db_file)):
            self.embeddings = np.load(os.path.join(cache_dir, db_file))
            if len(documents) == len(self.embeddings):
                return self.embeddings

        return self.build_embeddings(documents)

def verify_model(semantic_search: SemanticSearch):
    print(f"Model loaded: {semantic_search.model}")
    print(f"Max sequence length: {semantic_search.model.max_seq_length}")

def embed_text(text: str):
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def verify_embeddings():
    semantic_search = SemanticSearch()
    data = data_read("data/movies.json")
    documents = data["movies"]
    embeddings = semantic_search.load_or_create_embeddings(documents)

    print(f"Number of docs: {len(documents)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")

