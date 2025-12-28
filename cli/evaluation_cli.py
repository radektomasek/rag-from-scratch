import argparse

from search.hybrid_search import HybridSearch
from search.utils import data_read

def extract_retrieved_titles(results: list[dict]) -> list[str]:
    return list(map(lambda x: x["document"]["title"], results))

def calc_precision(retrieved_titles: list[str], relevant_titles: list[str]) -> float:
    return len(set(retrieved_titles).intersection(set(relevant_titles))) / len(retrieved_titles)

def calc_recall(retrieved_titles: list[str], relevant_titles: list[str]) -> float:
    return len(set(retrieved_titles).intersection(set(relevant_titles))) / len(relevant_titles)

def calc_f1_score(precision_score: float, recall_score: float) -> float:
    return 2 * (precision_score * recall_score) / (precision_score + recall_score)

def main():
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit
    rrf_k = 60

    movie_data = data_read("data/movies.json")
    hybrid_search = HybridSearch(movie_data["movies"])

    eval_data = data_read("data/golden_dataset.json")
    for test_case in eval_data["test_cases"]:
        query = test_case.get("query")
        relevant_docs = test_case.get("relevant_docs")
        retrieved_docs = hybrid_search.rrf_search(query, rrf_k, limit)
        retrieved_titles = extract_retrieved_titles(retrieved_docs[:limit])
        precision = calc_precision(retrieved_titles, relevant_docs)
        recall = calc_recall(retrieved_titles, relevant_docs)
        f1_score = calc_f1_score(precision, recall)

        print(f"- Query: {query}")
        print(f"  - Precision@{limit}: {precision:.4f}")
        print(f"  - Recall@{limit}: {recall:.4f}")
        print(f"  - F1 Score: {f1_score:.4f}")
        print(f"  - Retrieved: {retrieved_titles}")
        print(f"  - Relevant: {relevant_docs}")
        print("\n")

if __name__ == "__main__":
    main()