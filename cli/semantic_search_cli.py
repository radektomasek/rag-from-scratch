import argparse

from search.data_processing import data_read
from lib.chunked_semantic_search import ChunkedSemanticSearch
from lib.chunking import basic_chunking, semantic_chunking
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

    chunk_parser = subparsers.add_parser("chunk", help="Split the text into chunks")
    chunk_parser.add_argument("text", help="Specify text for the chunking")
    chunk_parser.add_argument("--chunk-size", type=int, nargs="?", default=200, help="The size of an individual chunk")
    chunk_parser.add_argument("--overlap", type=int, nargs="?", default=0, help="Specify an overlap of words in chunking")

    semantic_chunk_parser = subparsers.add_parser("semantic_chunk", help="Split the text into semantic chunks")
    semantic_chunk_parser.add_argument("text", help="Specify text for the semantic chunking")
    semantic_chunk_parser.add_argument("--max-chunk-size", type=int, nargs="?", default=4, help="The size of a chunk object")
    semantic_chunk_parser.add_argument("--overlap", type=int, nargs="?", default=0, help="Specify an overlap of words in chunking")

    subparsers.add_parser("embed_chunks", help="Prepare embeddings per chunk")

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

        case "chunk":
            text = args.text
            size = args.chunk_size
            overlap = args.overlap

            print(f"Chunking {len(text)} characters")
            for index, chunk in enumerate(basic_chunking(text, size, overlap), start=1):
                print(f"{index}. {chunk}")

        case "semantic_chunk":
            text = args.text
            size = args.max_chunk_size
            overlap = args.overlap

            print(f"Semantically chunking {len(text)} characters")
            for index, chunk in enumerate(semantic_chunking(text, size, overlap), start=1):
                print(f"{index}. {chunk}")

        case "embed_chunks":
            data = data_read("data/movies.json")
            documents = data["movies"]

            chunked_semantic_search = ChunkedSemanticSearch()
            embeddings = chunked_semantic_search.load_or_create_chunk_embeddings(documents)
            print(f"Generated {len(embeddings)} chunked embeddings")

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()