import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
print(f"Using key {api_key[:6]}...")

client = genai.Client(api_key=api_key)
model = "gemini-2.0-flash-001"
contents = "Why is Boot.dev such a great place to learn about RAG? Use one paragraph maximum."


results = client.models.generate_content(model=model, contents=contents)
print(results.text)
print(f"Prompt Tokens: {results.usage_metadata.prompt_token_count - 1}")
print(f"Response Tokens: {results.usage_metadata.candidates_token_count}")