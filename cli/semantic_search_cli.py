import argparse

from search.data_processing import data_read
from lib.semantic_search import (
    SemanticSearch,
    verify_model,
    embed_text,
    verify_embeddings,
    embed_query_text
)

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="Print the model information for the vector search")

    subparsers.add_parser("verify_embeddings", help="Verify the document embeddings")

    query_embedding_parser = subparsers.add_parser("embedquery", help="find data based on the embed query.")
    query_embedding_parser.add_argument("query", help="Query for the embedding search")

    embed_text_parser = subparsers.add_parser("embed_text", help="Get the embedding from a specified text")
    embed_text_parser.add_argument("text", type=str, help="Text for the embedding function")

    embed_search_parser = subparsers.add_parser("search", help="Search the result based on semantic vectors")
    embed_search_parser.add_argument("query", help="Query for the embedding search")
    embed_search_parser.add_argument("--limit", type=int, nargs="?", default=5, help="The number of element in result data")

    args = parser.parse_args()

    match args.command:
        case "verify":
            semantic_search = SemanticSearch()
            verify_model(semantic_search)
        case "embed_text":
            text = args.text
            embed_text(text)
        case "verify_embeddings":
            verify_embeddings()
        case "embedquery":
            query = args.query
            embed_query_text(query)
        case "search":
            limit = args.limit
            query = args.query

            semantic_search = SemanticSearch()
            data = data_read("data/movies.json")
            documents = data["movies"]

            semantic_search.load_or_create_embeddings(documents)
            results = semantic_search.search(query, limit)

            for index, element in enumerate(results, start=1):
                print(f"{index}. {element[1]['title']} (score: {element[0]:.4f})")
                print(f"   {element[1]['description'][:80]}\n")



        case _:
            parser.print_help()

if __name__ == "__main__":
    main()