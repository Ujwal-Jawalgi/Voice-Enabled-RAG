"""
tts.py — Sarvam AI Text-to-Speech integration.
"""

import logging
import httpx

from app.config import settings

logger = logging.getLogger(__name__)

SARVAM_TTS_URL = "https://api.sarvam.ai/text-to-speech"

# Global persistent client for connection pooling (Requirement 8)
_client: httpx.AsyncClient | None = None

def get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(
            timeout=10.0,
            limits=httpx.Limits(max_keepalive_connections=50, max_connections=100)
        )
    return _client

# Map our internal language strings to Sarvam BCP-47 codes
LANGUAGE_MAP = {
    "english": "en-IN",
    "hindi": "hi-IN",
    "kannada": "kn-IN",
    "punjabi": "pa-IN",
    "tamil": "ta-IN",
    "telugu": "te-IN",
    "marathi": "mr-IN",
    "bengali": "bn-IN",
    "gujarati": "gu-IN",
    "malayalam": "ml-IN",
    "odia": "or-IN",
    "urdu": "ur-IN"
}

async def generate_audio(text: str, language: str) -> str:
    """
    Sends text to Sarvam AI TTS and returns base64-encoded audio.
    Uses 'priya' for a soft female voice, which handles Indic languages well.
    """
    if not text.strip():
        return ""
        
    client = get_client()
    lang_code = LANGUAGE_MAP.get(language.lower(), "en-IN")
    
    payload = {
        "text": text,
        "language_code": lang_code,
        "speaker": "priya",
        "model": "bulbul:v3"
    }
    
    headers = {
        "api-subscription-key": settings.sarvam_api_key,
        "Content-Type": "application/json"
    }
    
    try:
        response = await client.post(
            SARVAM_TTS_URL,
            json=payload,
            headers=headers
        )
        response.raise_for_status()
        
        resp_data = response.json()
        return resp_data.get("audios", [""])[0]
    except Exception as e:
        logger.error("TTS generation failed for text '%s': %s", text[:20], e)
        # We don't want TTS failure to crash the pipeline, so return empty audio
        return ""
