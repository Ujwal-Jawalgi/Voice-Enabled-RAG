import asyncio
import json
import codecs
from app.pipeline.llm import llm_stream, llm_call

async def main():
    with open("hindi_prompt_dump.json", "r", encoding="utf-8") as f:
        prompt = json.load(f)
        
    with codecs.open("investigate_exact_prompt_out.md", "w", "utf-8") as f:
        f.write("# Exact Prompt Investigation\n\n")
        
        f.write("## llm_stream Output\n")
        tokens = []
        try:
            async for chunk in llm_stream(prompt):
                tokens.append(chunk)
                f.write(f"- Token: `{repr(chunk)}`\n")
        except Exception as e:
            f.write(f"- Exception: {e}\n")
            
        full_answer = "".join(tokens)
        f.write(f"\nFinal Streamed Answer: `{repr(full_answer)}`\n")
        
        f.write("\n## llm_call Output\n")
        try:
            resp, attempt = await llm_call(prompt)
            f.write(f"Answer: `{repr(resp)}` (Attempt {attempt})\n")
        except Exception as e:
            f.write(f"Exception: {e}\n")

if __name__ == "__main__":
    asyncio.run(main())
