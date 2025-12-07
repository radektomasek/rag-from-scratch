import json
import os

import numpy as np

from .chunking import semantic_chunking
from .semantic_search import SemanticSearch

cache_dir = './cache'
db_file = 'chunk_embeddings.npy'
metadata_json = 'chunk_metadata.json'

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
