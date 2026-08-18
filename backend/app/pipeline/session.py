import json
import logging
import redis
from typing import List, Dict

from app.config import settings

logger = logging.getLogger(__name__)

# Initialize redis client safely
redis_client = None
if settings.redis_url:
    try:
        redis_client = redis.Redis.from_url(settings.redis_url, socket_connect_timeout=2)
        redis_client.ping()
    except Exception:
        redis_client = None

def get_history(session_id: str) -> List[Dict[str, str]]:
    """Retrieve chat history for a given session."""
    if not redis_client or not session_id:
        return []
    
    try:
        data = redis_client.get(f"session:{session_id}")
        if data:
            return json.loads(data)
    except Exception:
        pass
    return []

def append_history(session_id: str, query: str, answer: str):
    """Append a new interaction to the session history, set TTL to 2 hours."""
    if not redis_client or not session_id:
        return
    
    try:
        history = get_history(session_id)
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": answer})
        
        # Keep only the last 10 turns (20 messages) to prevent context overflow
        if len(history) > 20:
            history = history[-20:]
            
        redis_client.setex(
            f"session:{session_id}",
            7200, # 2 hours in seconds
            json.dumps(history)
        )
    except Exception:
        pass
