"""
stt.py — Groq Whisper Speech-to-Text integration.
"""

import base64
import time
import logging
import httpx
from typing import Tuple

from app.config import settings

logger = logging.getLogger(__name__)

GROQ_STT_URL = "https://api.groq.com/openai/v1/audio/transcriptions"


# Global persistent client for connection pooling (Requirement 8)
_client = httpx.AsyncClient(
    timeout=10.0,
    limits=httpx.Limits(max_keepalive_connections=50, max_connections=100)
)

async def process_audio(audio_base64: str) -> Tuple[str, str, float]:
    """
    Sends base64-encoded audio to Groq Whisper STT.
    
    Returns:
        tuple of (transcript: str, language: str, stt_time_ms: float)
        
    Raises:
        ValueError: with a user-facing error message if transcription fails.
    """
    t0 = time.perf_counter()
    
    try:
        # Strip potential data URI prefix if frontend sends it (e.g., data:audio/wav;base64,...)
        if "," in audio_base64:
            audio_base64 = audio_base64.split(",")[1]
            
        audio_bytes = base64.b64decode(audio_base64)
    except Exception as e:
        logger.error("Failed to decode base64 audio: %s", e)
        raise ValueError("Failed to decode audio. Please try recording again.")

    try:
        files = {
            # Groq Whisper supports webm directly
            "file": ("recording.webm", audio_bytes, "audio/webm")
        }
        data = {
            "model": "whisper-large-v3",
            "response_format": "verbose_json"
        }
        headers = {
            "Authorization": f"Bearer {settings.groq_api_key}"
        }
        
        response = await _client.post(
            GROQ_STT_URL,
            files=files,
            data=data,
            headers=headers
        )
        response.raise_for_status()
        
        resp_data = response.json()
        transcript = resp_data.get("text", "")
        # Groq returns capitalized language name, we lower it for consistency
        language = resp_data.get("language", "auto").lower()
        
        t_total = (time.perf_counter() - t0) * 1000
        
        if not transcript:
            raise ValueError("No speech detected.")
            
        return transcript, language, t_total
            
    except httpx.HTTPStatusError as e:
        logger.error("Groq API error %d: %s", e.response.status_code, e.response.text)
        raise ValueError("Speech transcription service is currently unavailable. Please try again.")
    except Exception as e:
        logger.error("STT network error: %s", e)
        raise ValueError("Failed to connect to the transcription service. Please try again.")
