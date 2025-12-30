import os
import sys

from PIL import Image
from sentence_transformers import SentenceTransformer
from .semantic_search import SemanticSearch, cosine_similarity

class MultimodalSearch:
    def __init__(self, model_name: str = "clip-ViT-B-32", documents: list[dict] = []):
        self.model = SentenceTransformer(model_name)
        self.documents = documents
        self.texts = list(map(lambda x: f"{x["title"]}: {x["description"]}", self.documents))
        self.text_embeddings = self.model.encode(self.texts, show_progress_bar=True)

    def embed_image(self, image_path: str ):
        image = Image.open(image_path)
        image_embeddings = self.model.encode(image)
        return image_embeddings


    def search_with_image(self, image_path: str):
        image_embedding = self.embed_image(image_path)
        similarities = list(map(lambda x: cosine_similarity(x, image_embedding), self.text_embeddings))
        results = []
        for element in zip(self.documents, similarities):
            data = {
                "id": element[0]["id"],
                "title": element[0]["title"],
                "description": element[0]["description"],
                "similarity": element[1]
            }
            results.append(data)
        return sorted(results, key=lambda x: x["similarity"], reverse=True)[:5]

def verify_image_embedding(image_path: str):
    if not os.path.exists(image_path):
        print(f"The file: {image_path} doesn't exist")
        sys.exit(1)

    search = MultimodalSearch()
    embedding = search.embed_image(image_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")
