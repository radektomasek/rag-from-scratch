import json

def data_read(file_name: str):
    with open(file_name, "r") as file:
        return json.load(file)

def debug_rrf(original_query: str, enhanced_query: str, results: list[dict], reranked_results: list[dict]):
    print(f"Original query: {original_query}")
    print(f"Enhanced query: {enhanced_query}")
    print(f"Results: {results}")
    print(f"Reranked results: {reranked_results}")