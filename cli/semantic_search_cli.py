import argparse

from lib.semantic_search import (
    SemanticSearch,
    verify_model
)

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="Print the model information for the vector search")
    args = parser.parse_args()

    semantic_search = SemanticSearch()

    match args.command:
        case "verify":
            verify_model(semantic_search)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()