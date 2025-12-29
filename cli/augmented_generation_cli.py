import argparse

from llm.gemini_client import (
    augment_resuts_by_llm,
    summarize_results_by_llm,
    enhance_results_by_citations,
    question_answering_by_llm
)
from search.hybrid_search import HybridSearch
from search.data_processing import data_read


def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    llm_summariser_parser = subparsers.add_parser("summarize", help="Summarize the search response by LLM")
    llm_summariser_parser.add_argument("query", type=str, help="Search query")
    llm_summariser_parser.add_argument("--limit", type=int, nargs="?", default=5, help="Default limit is 5")

    citation_enhancement_parser = subparsers.add_parser("citations", help="Add citation to the search response")
    citation_enhancement_parser.add_argument("query", type=str, help="Search query")
    citation_enhancement_parser.add_argument("--limit", type=int, nargs="?", default=5, help="Default limit is 5")

    question_answering_parser = subparsers.add_parser("question", help="Perform QA (search + answer question)")
    question_answering_parser.add_argument("question", type=str, help="Question to ask")
    question_answering_parser.add_argument("--limit", type=int, nargs="?", default=5, help="Default limit is 5")

    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query
            k = 60
            limit = 5

            data = data_read("data/movies.json")
            hybrid_search = HybridSearch(data["movies"])

            search_results = hybrid_search.rrf_search(query, k, limit)
            augmented_result = augment_resuts_by_llm(query, search_results)

            print("Search Results:")
            for element in search_results:
                print(f"   - {element["document"]["title"]}")

            print("\nRAG Response:")
            print(augmented_result)

        case "summarize":
            query = args.query
            limit = args.limit
            k = 60

            data = data_read("data/movies.json")
            hybrid_search = HybridSearch(data["movies"])

            search_results = hybrid_search.rrf_search(query, k, limit)
            summary_result = summarize_results_by_llm(query, search_results)

            print("Search Results:")
            for element in search_results:
                print(f"   - {element["document"]["title"]}")

            print("\nLLM Summary:")
            print(summary_result)

        case "citations":
            query = args.query
            limit = args.limit
            k = 60

            data = data_read("data/movies.json")
            hybrid_search = HybridSearch(data["movies"])

            search_results = hybrid_search.rrf_search(query, k, limit)
            citation_result = enhance_results_by_citations(query, search_results)

            print("Search Results:")
            for element in search_results:
                print(f"   - {element["document"]["title"]}")

            print("\nLLM Answer:")
            print(citation_result)

        case "question":
            question = args.question
            limit = args.limit
            k = 60

            data = data_read("data/movies.json")
            hybrid_search = HybridSearch(data["movies"])

            search_results = hybrid_search.rrf_search(question, k, limit)
            answer = question_answering_by_llm(question, search_results)

            print("Search Results:")
            for element in search_results:
                print(f"   - {element["document"]["title"]}")

            print("\nAnswer:")
            print(answer)

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()