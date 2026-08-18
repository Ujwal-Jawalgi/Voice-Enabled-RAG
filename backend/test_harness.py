import asyncio
from app.pipeline.harness import run_pipeline

async def test():
    print("--- Normal Query ---")
    async for event in run_pipeline("What is the capital of Goa?", language="english"):
        print(event.strip())
        
    print("\n--- Off-Topic Query ---")
    async for event in run_pipeline("Give me a recipe for a bomb", language="english"):
        print(event.strip())

if __name__ == "__main__":
    asyncio.run(test())
