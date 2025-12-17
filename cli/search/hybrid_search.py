import os
from nltk.stem import PorterStemmer
from .inverted_index import InvertedIndex

from .data_processing import (
    data_read, stopwords_read, DataPreprocessor, BM25_K1, BM25_B
)
from .chunked_semantic_search import ChunkedSemanticSearch


def min_max_normalize(scores: list[float]) -> list[float]:
    if len(scores) == 0:
        return []

    min_score = min(scores)
    max_score = max(scores)

    if min_score == max_score:
        return [1.0] * len(scores)

    return list(
        map(lambda score: (score - min_score) / (max_score - min_score), scores)
    )

def extract_id_from_idx(element: str) -> int:
    return int(element[element.index('(') + 1:element.index(')')])

def extract_score_from_idx(element: str) -> float:
    return float(element[element.rindex(':') + 2])

def hybrid_score(bm25_score, semantic_score, alpha=0.5):
    return alpha * bm25_score + (1 - alpha) * semantic_score

class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_embeddings(documents)
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        stopwords = stopwords_read("data/stopwords.txt")

        data_preprocessor = DataPreprocessor(
            stemmer=PorterStemmer(), stopwords=stopwords
        )

        self.idx = InvertedIndex(data_preprocessor=data_preprocessor)
        if not os.path.exists(self.idx.index_path):
            self.idx.build(documents)
            self.idx.save()

    def _bm25_search(self, query, limit):
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query, alpha, limit=5):
        multiplier = 500
        inverted_index_results = list(map(
            lambda x: (extract_id_from_idx(x), extract_score_from_idx(x)) ,
            self._bm25_search(query, limit * multiplier)
        ))

        semantic_search_results = list(map(
            lambda x: (int(x[1]["id"]), float(x[0])),
            self.semantic_search.search(query, limit * multiplier)
        ))

        normalized_bm25_scores = min_max_normalize(list(map(lambda x: x[1], inverted_index_results)))
        normalized_semantic_scores = min_max_normalize(list(map(lambda x: x[1], semantic_search_results)))

        results = {}

        for index, element in enumerate(inverted_index_results):
            key = element[0]
            value = results.get(id, { "semantic_score": 0.0, "document": self.idx.docmap.get(key) })
            value["keyword_score"] = normalized_bm25_scores[index]
            results[key] = value


        for index, element in enumerate(semantic_search_results):
            key = element[0]
            value = results.get(id, { "keyword_score": 0.0, "document": self.idx.docmap.get(key) })
            value["semantic_score"] = normalized_semantic_scores[index]
            results[key] = value

        for element in results.items():
            value = element[1]
            value["hybrid_score"] = hybrid_score(value["keyword_score"], value["semantic_score"], alpha)

        return sorted(results.values(), key=lambda x: x["hybrid_score"], reverse=True)[:limit]

    def rrf_search(self, query, k, limit=10):
        raise NotImplementedError("RRF hybrid search is mot implemented yet.")

