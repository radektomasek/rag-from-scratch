import json
import os

import numpy as np

from .chunking import semantic_chunking
from .semantic_search import SemanticSearch, cosine_similarity

cache_dir = './cache'
db_file = 'chunk_embeddings.npy'
metadata_json = 'chunk_metadata.json'

SCORE_PRECISION = 4

class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def build_chunk_embeddings(self, documents):
        self.documents = documents

        for document in documents:
            doc_id = document.get("id")
            if not doc_id:
                raise ValueError("Missing 'id' in the document element")
            self.document_map[doc_id] = document

        chunks: list[str] = []
        chunks_metadata: list[dict] = []

        for document_index, document in enumerate(documents):
            description = document.get("description")
            if not description:
                continue

            chunks_elements = semantic_chunking(description, 4, 1)
            for chunk_index, chunk_element in enumerate(chunks_elements):
                chunks.append(chunk_element)
                metadata = {
                    "movie_idx": document_index,
                    "chunk_idx": chunk_index,
                    "total_chunks": len(chunks_elements)
                }
                chunks_metadata.append(metadata)

        self.chunk_embeddings = self.model.encode(chunks)
        self.chunk_metadata = chunks_metadata

        np.save(os.path.join(cache_dir, db_file), self.chunk_embeddings)

        with open(os.path.join(cache_dir, metadata_json), "w") as file:
            json.dump({
                "chunks": chunks_metadata,
                "total_chunks": len(chunks)
            }, file, indent=2 )

        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents

        for document in documents:
            doc_id = document.get("id")
            if not doc_id:
                raise ValueError("Missing 'id' in the document element")
            self.document_map[doc_id] = document

        if os.path.exists(os.path.join(cache_dir, db_file)) and os.path.exists(os.path.join(cache_dir, metadata_json)):
            self.chunk_embeddings = np.load(os.path.join(cache_dir, db_file))

            with open(os.path.join(cache_dir, metadata_json), "r") as file:
                data = json.load(file)
                self.chunk_metadata = data["chunks"]

            return self.chunk_embeddings

        return self.build_chunk_embeddings(documents)

    def search_chunks(self, query: str, limit: int = 10):
        query_embeddings = self.generate_embedding(query)
        chunks_scores = []

        for index, chunk_embedding in enumerate(self.chunk_embeddings):
            chunk_idx = self.chunk_metadata[index]["chunk_idx"]
            movie_idx = self.chunk_metadata[index]["movie_idx"]
            score = cosine_similarity(chunk_embedding, query_embeddings)
            chunks_scores.append({
                "chunk_idx": chunk_idx,
                "movie_idx": movie_idx,
                "score": score
            })

        movie_scores = {}
        for chunks_score in chunks_scores:
            movie_idx = chunks_score["movie_idx"]
            movie_score = movie_scores.get(movie_idx)
            if not movie_score:
                movie_scores[movie_idx] = chunks_score["score"]
            else:
                existing_score = movie_scores[movie_idx]
                new_score = chunks_score["score"]
                movie_scores[movie_idx] = new_score if new_score > existing_score else existing_score

        sorted_result = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)
        results = []
        for element in sorted_result[0:limit]:
            document = self.documents[element[0]]
            result = {
                "id": document["id"],
                "title": document["title"],
                "document": document["description"][:100],
                "score": round(element[1], SCORE_PRECISION),
                "metadata": {}
            }
            results.append(result)

        return results