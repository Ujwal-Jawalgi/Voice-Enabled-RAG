"""
sentence_buffer.py — Lightweight, deterministic sentence boundary detection.
"""

import re
import logging

logger = logging.getLogger(__name__)

class SentenceBuffer:
    def __init__(self):
        self.buffer = ""
        # Match ., ?, !, or Devanagari danda (।) as sentence enders.
        # Lookbehind prevents breaking on common abbreviations.
        self.split_pattern = re.compile(
            r'(?<=[.?!।])'                           # End punctuation
            r'(?<!\bMr\.)(?<!\bDr\.)(?<!\bMs\.)'     # Exclude common abbreviations
            r'(?<!\bvs\.)(?<!\be\.g\.)(?<!\bi\.e\.)' # Exclude common abbreviations
            r'(?<!\betc\.)'                          
            r'\s+'                                   # Followed by whitespace
        )

    def feed(self, text: str) -> list[str]:
        """Feed tokens into the buffer and return any completed sentences."""
        self.buffer += text
        
        sentences = []
        while True:
            match = self.split_pattern.search(self.buffer)
            if not match:
                break
                
            split_idx = match.end()
            sentence = self.buffer[:split_idx].strip()
            
            if sentence:
                sentences.append(sentence)
                
            self.buffer = self.buffer[split_idx:]
            
        return sentences

    def flush(self) -> str | None:
        """Return any trailing incomplete sentence when the stream ends."""
        sentence = self.buffer.strip()
        self.buffer = ""
        return sentence if sentence else None
