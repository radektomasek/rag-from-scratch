import argparse
import sys

from nltk.stem import PorterStemmer
from search.inverted_index import InvertedIndex
from search.data_processing import (
    data_read, stopwords_read, DataPreprocessor, BM25_K1, BM25_B
)

def load_data(db: InvertedIndex) -> None:
    try:
        db.load()
    except FileNotFoundError as error:
        print(error)
        sys.exit(1)

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    subparsers.add_parser("build", help="Extract data from input file to the internal cache")

    term_frequency_parser = subparsers.add_parser("tf", help="Find a term frequency for a document id")
    term_frequency_parser.add_argument("doc_id", type=int, help="Provide a document id to find a term frequency")
    term_frequency_parser.add_argument("search_phrase", type=str, help="Specify a search phrase you would like to find the occurrence for")

    idf_parser = subparsers.add_parser("idf", help="Calculate Inverse Document Frequency (idf) based on the search term/available data")
    idf_parser.add_argument("search_phrase", type=str, help="Specify a search phrase you would like to calculate the frequency for")

    tfidf_parser = subparsers.add_parser("tfidf", help="Calculate the 'Term and Inverse Document Frequency score for the specified keywords'")
    tfidf_parser.add_argument("doc_id", type=int, help="Provide a document id to find a TF_IDF score")
    tfidf_parser.add_argument("search_phrase", type=str, help="Specify a search phrase you would like to calculate the TF_IDF for")

    bm25_idf_parser = subparsers.add_parser("bm25idf", help="Get BM25 IDF score for a given term")
    bm25_idf_parser.add_argument("search_phrase", type=str, help="Term to get BM25 IDF score for")

    bm25_tf_parser = subparsers.add_parser("bm25tf", help="Get BM25 TF Score for a given document ID and term")
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("search_phrase", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument("k1", type=float, nargs="?", default=BM25_K1, help="Tunable BM25 K1 parameter")
    bm25_tf_parser.add_argument("b", type=float, nargs="?", default=BM25_B, help="Tunable BM25 b parameter")

    bm25search_parser = subparsers.add_parser("bm25search", help="Search movies using full BM25 scoring")
    bm25search_parser.add_argument("query", type=str, help="Search query")
    bm25search_parser.add_argument("--limit", type=int, nargs="?", default=5, help="Limit result to N")

    args = parser.parse_args()
    stopwords = stopwords_read("data/stopwords.txt")

    data_preprocessor = DataPreprocessor(
        stemmer=PorterStemmer(), stopwords=stopwords
    )
    inverted_index = InvertedIndex(data_preprocessor=data_preprocessor)

    match args.command:
        case "build":
            data = data_read("data/movies.json")
            inverted_index.build(data["movies"])
            inverted_index.save()

        case "search":
            load_data(inverted_index)

            search_query = args.query
            results = []
            for search_phrase in search_query.split():
                documents = inverted_index.index.get(data_preprocessor.transform(search_phrase), set())
                results.extend(list(documents))
                results = sorted(list(set(results)))

                if len(results) > 5:
                    results = results[0:5]
                    break
                else:
                    continue

            for doc_id in results:
                id = inverted_index.docmap[doc_id]["id"]
                title = inverted_index.docmap[doc_id]["title"]
                print(f"{id} - {title}")

        case "tf":
            load_data(inverted_index)

            doc_id = args.doc_id
            search_phrase = args.search_phrase
            result = inverted_index.get_tf(doc_id, search_phrase)
            print(result)

        case "idf":
            load_data(inverted_index)

            search_phrase = args.search_phrase
            result = inverted_index.get_idf(search_phrase)
            print(f"Inverse document frequency of '{search_phrase}': {result:.2f}")

        case "tfidf":
            load_data(inverted_index)

            doc_id = args.doc_id
            search_phrase = args.search_phrase
            result = inverted_index.get_tf(doc_id, search_phrase) * inverted_index.get_idf(search_phrase)
            print(f"TF-IDF score of '{search_phrase}' in document '{doc_id}': {result:.2f}")

        case "bm25tf":
            load_data(inverted_index)

            doc_id = args.doc_id
            search_phrase = args.search_phrase
            k1 = args.k1
            b = args.b
            result = inverted_index.get_bm25_tf(doc_id, search_phrase, k1, b)
            print(f"BM25 TF Score of '{search_phrase}' in document '{doc_id}': {result:.2f}")

        case "bm25idf":
            load_data(inverted_index)

            search_phrase = args.search_phrase
            result = inverted_index.get_bm25_idf(search_phrase)
            print(f"BM25 IDF score of '{search_phrase}': {result:.2f}")

        case "bm25search":
            load_data(inverted_index)
            query = args.query
            limit = args.limit

            result = inverted_index.bm25_search(query, limit)
            for element in result:
                print(element)

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()