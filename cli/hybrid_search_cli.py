import argparse
from time import sleep

from search.reranking import cross_encoder_rerank
from search.utils import data_read
from search.hybrid_search import HybridSearch, min_max_normalize
from llm.gemini_client import (
    query_spell_check_by_llm,
    query_rewrite_by_llm,
    query_expand_by_llm,
    calculate_rerank_score_by_llm,
    calculate_rerank_relevance_by_llm
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
    rrf_search_parser.add_argument("--rerank-method", type=str, choices=["individual", "batch", "cross_encoder"], help="Reranking method for adjusting the results by LLM")

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
            rerank_method = args.rerank_method
            original_limit = args.limit
            limit = original_limit * 5 if rerank_method else original_limit
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

            if rerank_method == "individual":
                for result in results:
                    result["rerank_score"] = calculate_rerank_score_by_llm(query, result["document"])
                    sleep(3)
                results = sorted(results, key=lambda x: x["rerank_score"], reverse=True)
            elif rerank_method == "batch":
                rerank_ids = calculate_rerank_relevance_by_llm(query, results)
                for result in results:
                    result["rerank_rank"] = rerank_ids.index(result["document"]["id"]) + 1 if result["document"]["id"] in rerank_ids else 0
                results = sorted(results, key=lambda x: x["rerank_rank"], reverse=False)
            elif rerank_method == "cross_encoder":
                pairs = [[query, f"{result['document'].get("title", "")} - {result['document']}"] for result in results]
                scores = cross_encoder_rerank(pairs)
                for result, score in zip(results, scores):
                    result["cross_encoder_score"] = score
                results = sorted(results, key=lambda x: x["cross_encoder_score"], reverse=True)

            for index, result in enumerate(results[:original_limit], start=1):
                rerank_score = result.get("rerank_score")
                rerank_rank = result.get("rerank_rank")
                cross_encoder_score = result.get("cross_encoder_score")
                print(f"{index}. {result["document"]["title"]}")
                if rerank_score:
                    print(f"   Rerank Score: {rerank_score:.4f}/10")
                if rerank_rank:
                    print(f"   Rerank Rank: {rerank_rank}")
                if cross_encoder_score:
                    print(f"   Cross Encoder Score: {cross_encoder_score:.4f}")
                print(f"   RRF Score: {result["rrf_score"]:4f}")
                print(f"   BM25 Rank: {result.get("bm25_rank", 0)}, Semantic Rank: {result.get("semantic_rank", 0)}")
                print(f"   {result["document"]["description"][:50]}...")
                print("\n")

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()