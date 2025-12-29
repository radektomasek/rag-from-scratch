import json
import os

from dotenv import load_dotenv
from google import genai

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")

client = genai.Client(api_key=api_key)

def is_float(value: str) -> bool:
    try:
        float(value)
        return True
    except:
        return False

def spell_check_prompt(query: str) -> str:
    return (
        f"""
Fix any spelling errors in this movie search query.
Only correct obvious typos. Don't change correctly spelled words.
Query: "{query}"
If no errors, return the original query.
Corrected:
        """.strip()
    )

def rewrite_query_prompt(query: str) -> str:
    return (
        f"""
Rewrite this movie search query to be more specific and searchable.

Original: "{query}"

Consider:
- Common movie knowledge (famous actors, popular films)
- Genre conventions (horror = scary, animation = cartoon)
- Keep it concise (under 10 words)
- It should be a google style search query that's very specific
- Don't use boolean logic

Examples:

- "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
- "movie about bear in london with marmalade" -> "Paddington London marmalade"
- "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

Rewritten query:
        """.strip()
    )

def expand_query_prompt(query: str) -> str:
    return (
        f"""
Expand this movie search query with related terms.

Add synonyms and related concepts that might appear in movie descriptions.
Keep expansions relevant and focused.
This will be appended to the original query.

Examples:

- "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
- "action movie with bear" -> "action thriller bear chase fight adventure"
- "comedy with bear" -> "comedy funny bear humor lighthearted"

Query: "{query}"
    """.strip()
    )

def rerank_basic_prompt(query: str, doc: dict) -> str:
    return (
        f"""
Rate how well this movie matches the search query.

Query: "{query}"
Movie: {doc.get("title", "")} - {doc.get("document", "")}

Consider:
- Direct relevance to query
- User intent (what they're looking for)
- Content appropriateness

Rate 0-10 (10 = perfect match).
Give me ONLY the number in your response, no other text or explanation.

Score:
        """.strip()
    )

def rerank_batch_prompt(query: str, doc_list_str: str) -> str:
    return (
        f"""
Rank these movies by relevance to the search query.

Query: "{query}"

Movies:
{doc_list_str}

Return ONLY the IDs in order of relevance (best match first). Return a valid JSON list, nothing else. For example:

[75, 12, 34, 2, 1]
        """.strip()
    )

def evaluation_batch_prompt(query: str, formatted_results: list[str]) -> str:
    return (
        f"""
Rate how relevant each result is to this query on a 0-3 scale:

Query: "{query}"

Results:
{chr(10).join(formatted_results)}

Scale:
- 3: Highly relevant
- 2: Relevant
- 1: Marginally relevant
- 0: Not relevant

Do NOT give any numbers out than 0, 1, 2, or 3.

Return ONLY the scores in the same order you were given the documents. Return a valid JSON list, nothing else. For example:

[2, 0, 3, 2, 0, 1]
""".strip()
    )

def augmented_generation_prompt(query: str, docs: list[dict]) -> str:
    return (
        f"""
Answer the question or provide information based on the provided documents. This should be tailored to Hoopla users. Hoopla is a movie streaming service.

Query: {query}

Documents:
{docs}

Provide a comprehensive answer that addresses the query:
""".strip()
    )

def summarize_results_prompt(query: str, docs: list[dict]) -> str:
    return (
        f"""
Provide information useful to this query by synthesizing information from multiple search results in detail.
The goal is to provide comprehensive information so that users know what their options are.
Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.
This should be tailored to Hoopla users. Hoopla is a movie streaming service.
Query: {query}
Search Results:
{docs}
Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:
    """
    )

def citation_adding_prompt(query: str, docs: list[dict]) -> str:
    return (
        f"""Answer the question or provide information based on the provided documents.

This should be tailored to Hoopla users. Hoopla is a movie streaming service.

If not enough information is available to give a good answer, say so but give as good of an answer as you can while citing the sources you have.

Query: {query}

Documents:
{docs}

Instructions:
- Provide a comprehensive answer that addresses the query
- Cite sources using [1], [2], etc. format when referencing information
- If sources disagree, mention the different viewpoints
- If the answer isn't in the documents, say "I don't have enough information"
- Be direct and informative

Answer:
    """
    )

def question_answering_prompt(question: str, docs: list[dict]) -> str:
    return (
        f"""
Answer the user's question based on the provided movies that are available on Hoopla.

This should be tailored to Hoopla users. Hoopla is a movie streaming service.

Question: {question}

Documents:
{docs}

Instructions:
- Answer questions directly and concisely
- Be casual and conversational
- Don't be cringe or hype-y
- Talk like a normal person would in a chat conversation

Answer:
    """
    )

def query_spell_check_by_llm(query: str):
    model = "gemini-2.0-flash-001"
    prompt = [spell_check_prompt(query)]

    result = client.models.generate_content(model=model, contents=prompt)
    return result.text

def query_rewrite_by_llm(query: str):
    model = "gemini-2.0-flash-001"
    prompt = [rewrite_query_prompt(query)]

    result = client.models.generate_content(model=model, contents=prompt)
    return result.text

def query_expand_by_llm(query: str):
    model = "gemini-2.0-flash-001"
    prompt = [expand_query_prompt(query)]

    result = client.models.generate_content(model=model, contents=prompt)
    return result.text

def calculate_rerank_score_by_llm(query: str, doc: dict):
    model = "gemini-2.0-flash-001"
    prompt = [rerank_basic_prompt(query, doc)]

    result = client.models.generate_content(model=model, contents=prompt)
    return float(result.text) if is_float(result.text) else None

def calculate_rerank_relevance_by_llm(query: str, docs: list[str]):
    model = "gemini-2.0-flash-001"
    doc_list_str = ", ".join(list(map(lambda x: json.dumps(x["document"]), docs)))

    prompt = [rerank_batch_prompt(query, doc_list_str)]
    result = client.models.generate_content(model=model, contents=prompt)

    return json.loads(result.text[7:-3]) or []

def evaluate_results_by_llm(query: str, results: list[dict]):
    model = "gemini-2.0-flash-001"
    formatted_results = list(
        map(
            lambda x:
                f"'title': {x["document"].get("title", "")},"
                f"'description': {x["document"].get("description", "")[:600]}",
            results
        )
    )

    prompt = [evaluation_batch_prompt(query, formatted_results)]
    result = client.models.generate_content(model=model, contents=prompt)

    return json.loads(result.text) or []

def augment_resuts_by_llm(query: str, docs: list[dict]):
    model = "gemini-2.0-flash-001"
    data = list(
        map(
            lambda x: {
                "title": x["document"]["title"],
                "description": x["document"]["description"]},
            docs)
    )
    prompt = [augmented_generation_prompt(query, data)]
    result = client.models.generate_content(model=model, contents=prompt)

    return result.text

def summarize_results_by_llm(query: str, docs: list[dict]):
    model = "gemini-2.0-flash-001"
    data = list(
        map(
            lambda x: {
                "title": x["document"]["title"],
                "description": x["document"]["description"]},
            docs)
    )
    prompt = [summarize_results_prompt(query, data)]
    result = client.models.generate_content(model=model, contents=prompt)

    return result.text

def enhance_results_by_citations(query: str, docs: list[dict]):
    model = "gemini-2.0-flash-001"
    data = list(
        map(
            lambda x: {
                "title": x["document"]["title"],
                "description": x["document"]["description"]},
            docs)
    )
    prompt = [citation_adding_prompt(query, data)]
    result = client.models.generate_content(model=model, contents=prompt)

    return result.text

def question_answering_by_llm(query: str, docs: list[dict]):
    model = "gemini-2.0-flash-001"
    data = list(
        map(
            lambda x: {
                "title": x["document"]["title"],
                "description": x["document"]["description"]},
            docs)
    )
    prompt = [question_answering_prompt(query, docs)]
    result = client.models.generate_content(model=model, contents=prompt)

    return result.text