#%% Imports
from pathlib import Path
import os

from dotenv import load_dotenv
from openai import OpenAI

#%% Environment

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

API_KEY = os.environ["OPENAI_API_KEY"]
MODEL = os.getenv("OPENAI_MODEL", "gpt-5-nano")

client = OpenAI(api_key=API_KEY)

#%% API connection test

response = client.responses.create(
    model=MODEL,
    input="Reply with exactly: API connection successful.",
    max_output_tokens=20,
)

print(response.output_text)

#%% Basic generation test

response = client.responses.create(
    model=MODEL,
    input="Reply in one short sentence: What is a perspective?",
    reasoning={"effort": "minimal"},
    max_output_tokens=500,
)

print(f"Model: {MODEL}")
print(f"Status: {response.status}")
print("Response:")
print(response.output_text)

print("\nUsage:")
print(response.usage)