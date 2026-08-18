import asyncio
import os
import sys
from groq import AsyncGroq
from dotenv import load_dotenv

load_dotenv()

async def test():
    api_key = os.environ.get('GROQ_API_KEY')
    if not api_key:
        print("ERROR: GROQ_API_KEY is not set. Please add it to your .env file.")
        sys.exit(1)
        
    try:
        # Initialize client inside the try block to catch instantiation errors
        client = AsyncGroq(api_key=api_key)
        response = await client.models.list()
        
        print(f"Successfully connected! Found {len(response.data)} models:")
        for model in response.data:
            print("-", model.id)
            
    except Exception as e:
        print(f"ERROR: {type(e).__name__} - {e}")

if __name__ == "__main__":
    asyncio.run(test())
