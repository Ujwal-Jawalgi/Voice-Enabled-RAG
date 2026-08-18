import httpx
from app.config import settings
print("Fetching Groq models...")
resp = httpx.get("https://api.groq.com/openai/v1/models", headers={"Authorization": f"Bearer {settings.groq_api_key}"})
print(resp.json())
