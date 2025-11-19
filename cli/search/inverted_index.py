import os
import pickle
import math
from collections import Counter
from functools import reduce

from .data_processing import DataPreprocessor, BM25_K1, BM25_B


def read_data(file_path: str) -> dict:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} doesn't exist")

    with open(file_path, "rb") as file:
        return pickle.load(file)

def write_data(file_path: str, data: dict) -> None:
    with open(file_path, "wb") as file:
        pickle.dump(data, file)

def format_bm25_search(index: int, element: tuple[int, float], metadata: dict) -> str:
    doc_id, score = element
    return f"{index}. ({doc_id}) {metadata['title']} - Score: {score:.2f}"

class InvertedIndex:

    def __init__(self, data_preprocessor: DataPreprocessor):
        self.index = {}
        self.docmap = {}
        self.doc_lengths = {}
        self.term_frequencies = {}
        self.data_preprocessor = data_preprocessor

    def __add_document(self, doc_id: int, text: str):
        tokens = text.split()
        self.doc_lengths[doc_id] = len(tokens)
        for token in tokens:
            key = token.lower()
            element = self.index.get(key, set())
            element.add(doc_id)
            self.index[key] = element
            counter = self.term_frequencies.get(doc_id, Counter())
            counter.update([key])
            self.term_frequencies[doc_id] = counter

    def get_documents(self, term: str):
        keys = self.index.get(term.lower(), set())
        return sorted(list(keys))

    def __get_avg_doc_length(self) -> float:
        num_of_documents = len(self.doc_lengths.keys())
        if num_of_documents == 0:
            return 0.0

        sum_of_lengths = reduce(lambda accumulator, current: accumulator + current, self.doc_lengths.values(), 0.0)
        return sum_of_lengths / num_of_documents

    def build(self, movies: list[dict]):
        for movie in movies:
            doc_id = movie['id']
            text = f"{movie['title']} {movie['description']}"
            self.docmap[doc_id] = movie
            self.__add_document(doc_id, self.data_preprocessor.transform(text))

    def extract_tokens(self, term: str) -> list[str]:
        search_term = self.data_preprocessor.transform(term)
        tokens = DataPreprocessor.tokenize(search_term)
        if len(tokens) == 0:
            raise Exception("[extract_tokens]: No token found!")

        return tokens

    def get_tf(self, doc_id: int, term: str) -> int:
        frequencies = self.term_frequencies.get(doc_id, Counter())
        tokens = self.extract_tokens(term)
        return frequencies[tokens[0]]

    def get_bm25_tf(self, doc_id: int, term: str, k1 = BM25_K1, b = BM25_B) -> float:
        term_frequency = self.get_tf(doc_id, term)
        doc_length = self.doc_lengths[doc_id]
        length_norm = 1 - b + b * (doc_length / self.__get_avg_doc_length())
        return term_frequency * (k1 + 1) / (term_frequency + k1 * length_norm)

    def get_idf(self, term: str) -> float:
        doc_count = len(self.docmap.keys())
        tokens = self.extract_tokens(term)
        term_doc_count = reduce(
            lambda accumulator, current: accumulator + (1 if current.get(tokens[0], 0) > 0 else 0),
            self.term_frequencies.values(),
            0
        )

        return math.log((doc_count + 1) / (term_doc_count + 1))

    def get_bm25_idf(self, term: str) -> float:
        doc_count = len(self.docmap.keys())
        tokens = self.extract_tokens(term)
        term_doc_count = reduce(
            lambda accumulator, current: accumulator + (1 if current.get(tokens[0], 0) > 0 else 0),
            self.term_frequencies.values(),
            0
        )

        return math.log((doc_count - term_doc_count + 0.5) / (term_doc_count + 0.5) + 1)

    def bm25(self, doc_id: int, term: str) -> float:
        return self.get_bm25_tf(doc_id, term) * self.get_bm25_idf(term)

    def bm25_search(self, query: str, limit: int):
        scores = {}
        tokens = self.extract_tokens(query)

        for token in tokens:
            for doc_id in list(self.index[token]):
                value = scores.get(doc_id, 0.0)
                scores[doc_id] = value + self.bm25(doc_id, token)

        sorted_scores = dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))
        result = list(map(lambda x: format_bm25_search(x[0], x[1], self.docmap[x[1][0]]), enumerate(list(sorted_scores.items())[:limit], start=1)))
        return result

    def load(self, path_name: str = '.'):
        cache_dir = os.path.join(path_name, "cache")
        self.index = read_data(os.path.join(cache_dir, "index.pkl"))
        self.docmap = read_data(os.path.join(cache_dir, "docmap.pkl"))
        self.doc_lengths = read_data(os.path.join(cache_dir, "doc_lengths.pkl"))
        self.term_frequencies = read_data(os.path.join(cache_dir, "term_frequencies.pkl"))

    def save(self, path_name: str = '.'):
        cache_dir = os.path.join(path_name, "cache")

        if not os.path.isdir(cache_dir):
            os.mkdir(cache_dir)

        write_data(os.path.join(cache_dir, "index.pkl"), self.index)
        write_data(os.path.join(cache_dir, "docmap.pkl"), self.docmap)
        write_data(os.path.join(cache_dir, "doc_lengths.pkl"), self.doc_lengths)
        write_data(os.path.join(cache_dir, "term_frequencies.pkl"), self.term_frequencies)
