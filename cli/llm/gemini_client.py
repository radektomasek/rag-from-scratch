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