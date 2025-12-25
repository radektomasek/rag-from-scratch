import argparse

from search.utils import data_read
from search.hybrid_search import HybridSearch, min_max_normalize
from llm.gemini_client import (
    query_spell_check_by_llm,
    query_rewrite_by_llm,
    query_expand_by_llm
)

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser("normalize", help="Normalize the data")
    normalize_parser.add_argument("scores", type=float, nargs="+", help="Specify the scores for normalization")

    weighted_search_parser = subparsers.add_parser("weighted-search", help="Combine keyword and semantic search results")
    weighted_search_parser.add_argument("query", type=str, help="Search query")
    weighted_search_parser.add_argument("--alpha", type=float, nargs="?", default=0.5, help="Weightening factor for keyword/semantic search results")
    weighted_search_parser.add_argument("--limit", type=int, nargs="?", default=5, help="Default limit is 5")

    rrf_search_parser = subparsers.add_parser("rrf-search", help="RRF Hybrid Search")
    rrf_search_parser.add_argument("query", type=str, help="Search query")
    rrf_search_parser.add_argument("-k", type=int, nargs="?", default=60, help="The weight consideration factor constant")
    rrf_search_parser.add_argument("--limit", type=int, nargs="?", default=5, help="Default limit is 5")
    rrf_search_parser.add_argument("--enhance", type=str, choices=["spell", "rewrite", "expand"], help="Query enhancement method")

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

            for index, result in enumerate(results, start=1):
                print(f"{index}. ${result["document"]["title"]}")
                print(f"   Hybrid Score: {result['hybrid_score']:.4f}")
                print(f"   BM25: {result['keyword_score']:.4f}, Semantic: {result['semantic_score']:.4f}")
                print(f"   {result["document"]["description"][:50]}...")
                print("\n")

        case "rrf-search":
            query = args.query
            k = args.k
            limit = args.limit
            enhance = args.enhance

            enhanced_query = None

            if enhance:
                if enhance == "spell":
                    enhanced_query = query_spell_check_by_llm(query)
                    if query != enhanced_query:
                        print(f"Enhanced query ({enhance}): '{query}' -> '{enhanced_query}'\n")
                elif enhance == "rewrite":
                    enhanced_query = query_rewrite_by_llm(query)
                    if query != enhanced_query:
                        print(f"Enhanced query ({enhance}): '{query}' -> '{enhanced_query}'\n")
                elif enhance == "expand":
                    enhanced_query = query_expand_by_llm(query)
                    if query != enhanced_query:
                        print(f"Enhanced query ({enhance}): '{query}' -> '{enhanced_query}'\n")

            query = enhanced_query or query
            data = data_read("data/movies.json")
            hybrid_search = HybridSearch(data["movies"])

            results = hybrid_search.rrf_search(query, k, limit)

            for index, result in enumerate(results, start=1):
                print(f"{index}. {result["document"]["title"]}")
                print(f"   RRF Score: {result["rrf_score"]:4f}")
                print(f"   BM25 Rank: {result.get("bm25_rank", 0)}, Semantic Rank: {result.get("semantic_rank", 0)}")
                print(f"   {result["document"]["description"][:50]}...")
                print("\n")

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()