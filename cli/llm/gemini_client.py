import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")

client = genai.Client(api_key=api_key)

def spell_check_prompt(query: str):
    return (
        f"""
            Fix any spelling errors in this movie search query.
            Only correct obvious typos. Don't change correctly spelled words.
            Query: "{query}"
            If no errors, return the original query.
            Corrected:
        """.strip()
    )

def query_spell_check_by_llm(query: str):
    model = "gemini-2.0-flash-001"
    prompt = [spell_check_prompt(query)]

    result = client.models.generate_content(model=model, contents=prompt)
    return result.text
