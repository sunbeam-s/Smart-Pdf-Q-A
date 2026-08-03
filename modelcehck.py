import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

# Initialize client with your API key
client = genai.Client(api_key=os.getenv("GENAI_API_KEY"))

print("--- Text Generation Models ---")
for m in client.models.list():
    if "generateContent" in getattr(m, "supported_actions", []):
        print(f"Model Name: {m.name}")

print("\n--- Embedding Models ---")
for m in client.models.list():
    if "embedContent" in getattr(m, "supported_actions", []):
        print(f"Model Name: {m.name}")