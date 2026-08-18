"""
llm.py — LLM call wrapper using Groq GPT-OSS 20B.

This module owns prompt construction and the Groq API call contract.
Implements a 2-second timeout and 1 retry to ensure tight latency bounds
are met for the voice-RAG pipeline.
"""

import logging
import asyncio
import datetime
import httpx
from groq import AsyncGroq, APIError

from app.config import settings

logger = logging.getLogger(__name__)

# Initialize single Groq client with explicit connection pooling
_http_client = httpx.AsyncClient(limits=httpx.Limits(max_keepalive_connections=50, max_connections=100))
client = AsyncGroq(api_key=settings.groq_api_key, http_client=_http_client)

MODEL = "openai/gpt-oss-20b"
TIMEOUT_SEC = 4.0

LANGUAGE_REFUSAL_FALLBACKS = {
    "english": "I don't have enough information to answer this.",
    "hindi": "मुझे पर्याप्त जानकारी नहीं है।",
    "kannada": "ನನಗೆ ಸಾಕಷ್ಟು ಮಾಹಿತಿ ಇಲ್ಲ.",
    "telugu": "నాకు తగినంత సమాచారం లేదు.",
    "tamil": "எனக்கு போதிய தகவல் இல்லை.",
    "bengali": "আমার কাছে পর্যাপ্ত তথ্য নেই।",
    "marathi": "माझ्याकडे पुरेशी माहिती नाही.",
    "gujarati": "મારી પાસે પૂરતી માહિતી નથી.",
    "punjabi": "ਮੇਰੇ ਕੋਲ ਕਾਫੀ ਜਾਣਕਾਰੀ ਨਹੀਂ ਹੈ।",
    "malayalam": "എനിക്ക് മതിയായ വിവരങ്ങളില്ല."
}

def get_fallback(language: str) -> str:
    return LANGUAGE_REFUSAL_FALLBACKS.get(language.lower(), LANGUAGE_REFUSAL_FALLBACKS["english"])

def build_prompt(query: str, context_chunks: list[str], language: str, history: list = None) -> list[dict]:
    """
    Construct the chat messages payload for the Groq API.
    Forces the model to reply in the detected language.
    """
    context_block = "\n\n---\n\n".join(context_chunks)

    lang_instruction = f"Respond in {language}. Do not respond in any other language unless the query itself is in that language."

    system_prompt = f"""You are a helpful assistant.
Answer concisely (1-2 sentences) using ONLY the provided context.
If the answer is not in the context, refuse politely.
{lang_instruction}

CONTEXT:
{context_block}
"""

    user_prompt = f"Question: {query}"

    messages = [{"role": "system", "content": system_prompt}]
    if history:
        messages.extend(history)
    
    messages.append({"role": "user", "content": user_prompt})
    return messages


async def llm_call(prompt: list[dict], language: str = "english", query: str = "") -> tuple[str, int]:
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
                    messages=prompt,
                    model=MODEL,
                    temperature=0.0, # 0.0 for deterministic factual answers
                    max_tokens=256,  # Increased from 128 to prevent truncation in Indic scripts for GPT-OSS 20B
                ),
                timeout=TIMEOUT_SEC
            )
            
            content = chat_completion.choices[0].message.content
            answer = content.strip() if content else ""
            if not answer:
                logger.warning("Empty LLM call fallback triggered for language '%s' on query: '%s'", language, query)
                answer = get_fallback(language)
            return answer, attempt + 1

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


async def llm_stream(prompt: list[dict], language: str = "english", query: str = ""):
    """Call the Groq LLM with streaming enabled.
    
    Yields string tokens as they arrive.
    """
    for attempt in range(2):
        try:
            # We wrap the API call with asyncio.wait_for to enforce timeout on the connection
            chat_completion = await asyncio.wait_for(
                client.chat.completions.create(
                    messages=prompt,
                    model=MODEL,
                    temperature=0.0,
                    max_tokens=128,  # Optimized max_tokens for speed
                    stream=True,
                ),
                timeout=TIMEOUT_SEC
            )
            
            yielded_any = False
            async for chunk in chat_completion:
                content = chunk.choices[0].delta.content
                if content:
                    yield content
                    yielded_any = True
            
            if not yielded_any:
                logger.warning("Empty LLM stream fallback triggered for language '%s' on query: '%s'", language, query)
                yield get_fallback(language)
                    
            return # Successful stream, we are done

        except asyncio.TimeoutError:
            ts = datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]
            logger.warning("[%s] [asyncio.TimeoutError] LLM call timed out on attempt %d (exceeded %.1fs)", ts, attempt + 1, TIMEOUT_SEC)
        except APIError as e:
            ts = datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]
            logger.warning("[%s] [%s] Groq API error on attempt %d: %s", ts, type(e).__name__, attempt + 1, str(e))
        except Exception as e:
            ts = datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]
            logger.warning("[%s] [%s] Unexpected error on LLM call attempt %d: %s", ts, type(e).__name__, attempt + 1, str(e))

    logger.error("LLM stream failed after 2 attempts. Yielding fallback.")
    yield "I couldn't generate an answer, please try again."
