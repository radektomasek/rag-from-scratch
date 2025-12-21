import os
from nltk.stem import PorterStemmer
from .inverted_index import InvertedIndex

from .data_processing import (
    data_read, stopwords_read, DataPreprocessor, BM25_K1, BM25_B
)
from .chunked_semantic_search import ChunkedSemanticSearch

multiplier = 500

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

def rrf_score(rank, k=60):
    return 1 / (k + rank)

def extract_id_from_idx(element: str) -> int:
    return int(element[element.index('(') + 1:element.index(')')])

def extract_score_from_idx(element: str) -> float:
    return float(element[element.rindex(':') + 1 :].strip())

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
        inverted_index_results = list(map(
            lambda x: (extract_id_from_idx(x), extract_score_from_idx(x)) ,
            self._bm25_search(query, limit * multiplier)
        ))

        semantic_search_results = list(map(
            lambda x: (int(x["id"]), float(x["score"])),
            self.semantic_search.search_chunks(query, limit * multiplier)
        ))

        normalized_bm25_scores = min_max_normalize(list(map(lambda x: x[1], inverted_index_results)))
        normalized_semantic_scores = min_max_normalize(list(map(lambda x: x[1], semantic_search_results)))

        results = {}

        for index, element in enumerate(inverted_index_results):
            key = element[0]
            value = results.get(key, { "semantic_score": 0.0, "document": self.idx.docmap.get(key) })
            value["keyword_score"] = normalized_bm25_scores[index]
            results[key] = value

        for index, element in enumerate(semantic_search_results):
            key = element[0]
            value = results.get(key, { "keyword_score": 0.0, "document": self.idx.docmap.get(key) })
            value["semantic_score"] = normalized_semantic_scores[index]
            results[key] = value

        for element in results.items():
            value = element[1]
            value["hybrid_score"] = hybrid_score(value["keyword_score"], value["semantic_score"], alpha)

        return sorted(results.values(), key=lambda x: x["hybrid_score"], reverse=True)[:limit]

    def rrf_search(self, query, k, limit=10):
        inverted_index_results = list(map(
            lambda x: (extract_id_from_idx(x), extract_score_from_idx(x)) ,
            self._bm25_search(query, limit * multiplier)
        ))

        semantic_search_results = list(map(
            lambda x: (int(x["id"]), float(x["score"])),
            self.semantic_search.search_chunks(query, limit * multiplier)
        ))

        results = {}

        for index, element in enumerate(sorted(inverted_index_results, key=lambda x: x[1], reverse=True), start=1):
            key = element[0]
            value = results.get(key, { "document": self.idx.docmap.get(key) })
            value["bm25_rank"] = index
            results[key] = value

        for index, element in enumerate(sorted(semantic_search_results, key=lambda x: x[1], reverse=True), start=1):
            key = element[0]
            value = results.get(key, {"document": self.idx.docmap.get(key)})
            value["semantic_rank"] = index
            results[key] = value

        for element in results.items():
            value = element[1]

            rrf_bm25 = 0.0
            rrf_semantic = 0.0

            bm25_rank = value.get("bm25_rank")
            semantic_rank = value.get("semantic_rank")

            if bm25_rank:
                rrf_bm25 = rrf_score(bm25_rank, k)

            if semantic_rank:
                rrf_semantic = rrf_score(semantic_rank, k)

            rrf_combined = rrf_bm25 + rrf_semantic
            value["rrf_score"] = rrf_combined

        return sorted(results.values(), key=lambda x: x["rrf_score"], reverse=True)[:limit]
