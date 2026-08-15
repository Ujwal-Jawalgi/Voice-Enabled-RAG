"""
llm.py — LLM call wrapper using Groq Llama 3.1 8B Instant.

This module owns prompt construction and the Groq API call contract.
Implements a 2-second timeout and 1 retry to ensure tight latency bounds
are met for the voice-RAG pipeline.
"""

import logging
import asyncio
import datetime
from groq import AsyncGroq, APIError

from app.config import settings

logger = logging.getLogger(__name__)

# Initialize single Groq client
client = AsyncGroq(api_key=settings.groq_api_key)

MODEL = "llama-3.1-8b-instant"
TIMEOUT_SEC = 4.0


def build_prompt(query: str, context_chunks: list[str], language: str) -> str:
    """Construct the grounded RAG prompt from query + retrieved context.

    The prompt instructs the LLM to:
    1. Answer ONLY from the provided context.
    2. Say it doesn't know if the context doesn't support an answer.
    3. Keep the answer concise.
    4. Respond in the requested language (Hindi/Kannada/English).
    """
    context_block = "\n\n---\n\n".join(context_chunks)

    lang_instruction = "Answer ONLY in English."
    if language.lower() == "hindi":
        lang_instruction = "Answer ONLY in Hindi (Devanagari script)."
    elif language.lower() == "kannada":
        lang_instruction = "Answer ONLY in Kannada script."

    return f"""You are a multilingual information retrieval assistant. 

CRITICAL INSTRUCTIONS:
1. Answer the user's question using ONLY the context passages provided below. 
2. If the context does not contain enough information to answer, say "I don't have enough information to answer this question." Do not attempt to guess or use outside knowledge.
3. Keep your answer concise (2-3 sentences).
4. {lang_instruction}

Context passages:
{context_block}

Question: {query}

Answer:"""


async def llm_call(prompt: str) -> tuple[str, int]:
    """Call the Groq LLM with the given prompt.

    Enforces a strict timeout and allows 1 retry on failure.
    On complete failure, returns a typed fallback response and the total attempts used.
    """
    fallback_response = "I couldn't generate an answer, please try again."
    
    for attempt in range(2):
        try:
            # We wrap the API call with asyncio.wait_for to enforce the 2s timeout
            chat_completion = await asyncio.wait_for(
                client.chat.completions.create(
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                    model=MODEL,
                    temperature=0.0, # 0.0 for deterministic factual answers
                    max_tokens=128,  # Keep answers concise and fast
                ),
                timeout=TIMEOUT_SEC
            )
            
            content = chat_completion.choices[0].message.content
            return (content.strip() if content else ""), attempt + 1

        except asyncio.TimeoutError:
            ts = datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]
            logger.warning("[%s] [asyncio.TimeoutError] LLM call timed out on attempt %d (exceeded %.1fs)", ts, attempt + 1, TIMEOUT_SEC)
        except APIError as e:
            ts = datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]
            logger.warning("[%s] [%s] Groq API error on attempt %d: %s", ts, type(e).__name__, attempt + 1, str(e))
        except Exception as e:
            ts = datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]
            logger.warning("[%s] [%s] Unexpected error on LLM call attempt %d: %s", ts, type(e).__name__, attempt + 1, str(e))

    logger.error("LLM call failed after 2 attempts. Returning fallback.")
    return fallback_response, 2
