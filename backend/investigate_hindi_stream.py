import asyncio
import codecs
from app.pipeline.llm import build_prompt, llm_stream, llm_call

async def main():
    with codecs.open("investigate_hindi_stream_out.md", "w", "utf-8") as f:
        f.write("# Hindi Stream Investigation\n\n")
        
        query_text = "गोवा की राजधानी क्या है?"
        language = "hindi"
        context_chunks = []
        prompt = build_prompt(query_text, context_chunks, language)
        
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

if __name__ == "__main__":
    asyncio.run(main())
