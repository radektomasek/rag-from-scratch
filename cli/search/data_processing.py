import json
import string
from nltk.stem import PorterStemmer

BM25_K1 = 1.5
BM25_B = 0.75

def data_read(file_name: str) -> dict[str, list[dict]]:
    with open(file_name, "r") as file:
        return json.load(file)

def stopwords_read(file_name: str) -> list[str]:
    with open(file_name, "r") as file:
        return file.read().splitlines()

class DataPreprocessor:
    def __init__(self, stemmer: PorterStemmer, stopwords: list[str] = None):
        self.stopwords = set(stopwords or [])
        self.stemmer = stemmer

    @staticmethod
    def lower(phrase: str) -> str:
        return phrase.lower()

    @staticmethod
    def tokenize(phrase: str) -> list[str]:
        return DataPreprocessor.remove_whitespace(phrase.split())

    @staticmethod
    def remove_punctuation(phrase: str) -> str:
        return phrase.translate(str.maketrans("", "", string.punctuation))

    @staticmethod
    def remove_whitespace(data: list[str]) -> list[str]:
        return list(filter(lambda x: x not in string.whitespace, data))

    def remove_stop_words(self, phrase: str) -> str:
        return " ".join(list(filter(lambda x: x not in self.stopwords, phrase.split())))

    def stem_words(self, phrase: str) -> str:
        return " ".join(list(map(lambda x: self.stemmer.stem(x), phrase.split())))

    def transform(self, phrase):
        phrase = self.lower(phrase)
        phrase = self.remove_punctuation(phrase)
        phrase = self.remove_stop_words(phrase)
        phrase = self.stem_words(phrase)
        return phrase