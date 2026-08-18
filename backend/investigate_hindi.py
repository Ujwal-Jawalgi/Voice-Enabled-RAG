import asyncio
import json
import codecs
from app.pipeline.llm import build_prompt, llm_stream, llm_call

async def main():
    with codecs.open("investigate_hindi_out.md", "w", "utf-8") as f:
        f.write("# Hindi Investigation\n\n")
        
        f.write("## 1 & 2. Exact Final Prompt\n")
        query_text = "गोवा की राजधानी क्या है?"
        language = "hindi"
        context_chunks = []
        prompt = build_prompt(query_text, context_chunks, language)
        f.write("```json\n")
        f.write(json.dumps(prompt, indent=2, ensure_ascii=False) + "\n")
        f.write("```\n\n")
        
        f.write("## 3. Testing Same Query 4 Times (Refusal Path)\n")
        for i in range(4):
            resp, attempt = await llm_call(prompt)
            f.write(f"- Test {i+1} Output (Attempt {attempt}): `{repr(resp)}`\n")

        f.write("\n## 4. Testing Answerable Hindi Query\n")
        answerable_query = "गोवा की राजधानी क्या है?"
        valid_context = [
            "Panaji is the capital of the Indian state of Goa and the headquarters of North Goa district.",
            "Goa is a state on the southwestern coast of India within the Konkan region."
        ]
        prompt_answerable = build_prompt(answerable_query, valid_context, language)
        resp_ans, attempt_ans = await llm_call(prompt_answerable)
        f.write(f"- Answerable Output (Attempt {attempt_ans}): `{repr(resp_ans)}`\n")

if __name__ == "__main__":
    asyncio.run(main())
