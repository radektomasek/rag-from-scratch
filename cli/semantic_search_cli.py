import argparse

from lib.semantic_search import (
    SemanticSearch,
    verify_model,
    embed_text,
    verify_embeddings
)

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="Print the model information for the vector search")

    subparsers.add_parser("verify_embeddings", help="Verify the document embeddings")

    embed_text_parser = subparsers.add_parser("embed_text", help="Get the embedding from a specified text")
    embed_text_parser.add_argument("text", type=str, help="Text for the embedding function")

    args = parser.parse_args()

    semantic_search = SemanticSearch()

    match args.command:
        case "verify":
            verify_model(semantic_search)
        case "embed_text":
            text = args.text
            embed_text(text)
        case "verify_embeddings":
            verify_embeddings()
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()