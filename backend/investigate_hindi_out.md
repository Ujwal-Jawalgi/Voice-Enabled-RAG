# Hindi Investigation

## 1 & 2. Exact Final Prompt
```json
[
  {
    "role": "system",
    "content": "You are a multilingual information retrieval assistant. \n\nCRITICAL INSTRUCTIONS:\n1. Answer the user's question using ONLY the context passages provided below. \n2. If the context does not contain enough information to answer, state that you do not have enough information to answer. Do not attempt to guess or use outside knowledge.\n3. Keep your answer EXTREMELY concise (1-2 sentences maximum). Do not add any pleasantries, conversational filler, or rambling.\n4. Respond in hindi. Do not respond in any other language unless the query itself is in that language.\n\nContext passages:\n"
  },
  {
    "role": "user",
    "content": "Question: गोवा की राजधानी क्या है?"
  }
]
```

## 3. Testing Same Query 4 Times (Refusal Path)
- Test 1 Output (Attempt 1): `'मुझे पर्याप्त जानकारी नहीं है।'`
- Test 2 Output (Attempt 1): `'मुझे पर्याप्त जानकारी नहीं है।'`
- Test 3 Output (Attempt 1): `'मुझे पर्याप्त जानकारी नहीं है।'`
- Test 4 Output (Attempt 1): `'मुझे पर्याप्त जानकारी नहीं है।'`

## 4. Testing Answerable Hindi Query
- Answerable Output (Attempt 1): `'गोवा की राजधानी पनाजी है।'`
