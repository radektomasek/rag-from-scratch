import argparse

from search.utils import data_read
from search.hybrid_search import HybridSearch, min_max_normalize

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser("normalize", help="Normalize the data")
    normalize_parser.add_argument("scores", type=float, nargs="+", help="Specify the scores for normalization")

    weighted_search_parser = subparsers.add_parser("weighted-search", help="Combine keyword and semantic search results")
    weighted_search_parser.add_argument("query", type=str, help="Search query")
    weighted_search_parser.add_argument("--alpha", type=float, nargs="?", default=0.5, help="Weightening factor for keyword/semantic search results")
    weighted_search_parser.add_argument("--limit", type=int, nargs="?", default=5, help="Default limit is 5")

    args = parser.parse_args()

    match args.command:
        case "normalize":
            scores = args.scores
            results = min_max_normalize(scores)
            for score in results:
                print(f"* {score:.4f}")

        case "weighted-search":
            query = args.query
            alpha = args.alpha
            limit = args.limit

            data = data_read("data/movies.json")
            hybrid_search = HybridSearch(data["movies"])

            results = hybrid_search.weighted_search(query, alpha, limit)

            for result in results:
                print(result["document"]["title"])





        case _:
            parser.print_help()

if __name__ == "__main__":
    main()