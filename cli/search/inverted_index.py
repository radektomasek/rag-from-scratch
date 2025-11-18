import os
import pickle
import math
from collections import Counter
from functools import reduce

from .data_processing import DataPreprocessor


def read_data(file_path: str) -> dict:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} doesn't exist")

    with open(file_path, "rb") as file:
        return pickle.load(file)

def write_data(file_path: str, data: dict) -> None:
    with open(file_path, "wb") as file:
        pickle.dump(data, file)

class InvertedIndex:

    def __init__(self, data_preprocessor: DataPreprocessor):
        self.index = {}
        self.docmap = {}
        self.term_frequencies = {}
        self.data_preprocessor = data_preprocessor

    def __add_document(self, doc_id: int, text):
        tokens = text.split()
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
        if len(tokens) > 1:
            raise Exception("[extract_tokens]: Only one token is allowed")

        return tokens

    def get_tf(self, doc_id: int, term: str) -> int:
        frequencies = self.term_frequencies.get(doc_id, Counter())
        tokens = self.extract_tokens(term)
        return frequencies[tokens[0]]

    def get_idf(self, term: str) -> float:
        doc_count = len(self.docmap.keys())
        tokens = self.extract_tokens(term)
        term_doc_count = reduce(
            lambda accumulator, current: accumulator + (1 if current.get(tokens[0], 0) > 0 else 0),
            self.term_frequencies.values(),
            0
        )

        return math.log((doc_count + 1) / (term_doc_count + 1))

    def load(self, path_name: str = '.'):
        cache_dir = os.path.join(path_name, "cache")
        self.index = read_data(os.path.join(cache_dir, "index.pkl"))
        self.docmap = read_data(os.path.join(cache_dir, "docmap.pkl"))
        self.term_frequencies = read_data(os.path.join(cache_dir, "term_frequencies.pkl"))

    def save(self, path_name: str = '.'):
        cache_dir = os.path.join(path_name, "cache")

        if not os.path.isdir(cache_dir):
            os.mkdir(cache_dir)

        write_data(os.path.join(cache_dir, "index.pkl"), self.index)
        write_data(os.path.join(cache_dir, "docmap.pkl"), self.docmap)
        write_data(os.path.join(cache_dir, "term_frequencies.pkl"), self.term_frequencies)
