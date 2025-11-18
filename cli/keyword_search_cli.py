import argparse
import sys

from nltk.stem import PorterStemmer
from search.inverted_index import InvertedIndex
from search.data_processing import (
    data_read, stopwords_read, DataPreprocessor
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

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()