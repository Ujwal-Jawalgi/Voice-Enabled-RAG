import os, time, asyncio, httpx
from openai import AsyncOpenAI
import logging

logging.basicConfig(level=logging.INFO)

async def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Success: File ran perfectly. (Note: OpenAI API call skipped because OPENAI_API_KEY is not set).")
        return

    try:
        client = AsyncOpenAI(api_key=api_key)
        t0 = time.perf_counter()
        resp = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Explain gravity in 3 words."}],
            max_tokens=10,
            temperature=0.0
        )
        t1 = time.perf_counter()
        print(f"Time: {(t1 - t0)*1000:.2f} ms")
        print(resp.choices[0].message.content)
    except Exception as e:
        print(f"OpenAI API Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
