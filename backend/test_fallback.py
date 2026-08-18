import urllib.request
import json
import codecs

url = 'http://127.0.0.1:8000/query'
headers = {
    'Content-Type': 'application/json'
}

queries = [
    ("Hindi", "गोवा की राजधानी क्या है?"),
    ("Kannada", "ಕೊಕಾ-ಕೋಲಾದ ರಹಸ್ಯ ಪಾಕವಿಧಾನವೇನು?"),
    ("Tamil", "கோகோ-கோலாவின் ரகசிய செய்முறை என்ன?")
]

def test_query(lang, text, out_file):
    print(f"Testing {lang}...")
    data = json.dumps({'text': text}).encode('utf-8')
    req = urllib.request.Request(url, data=data, headers=headers)
    
    try:
        response = urllib.request.urlopen(req)
        body = response.read().decode('utf-8')
        
        final_data = None
        for line in body.split("\n"):
            if line.startswith("data: "):
                try:
                    parsed = json.loads(line[6:])
                    if parsed.get("type") == "final":
                        final_data = parsed.get("response")
                except json.JSONDecodeError:
                    pass
        
        with codecs.open(out_file, "a", "utf-8") as f:
            f.write(f"### {lang}\n")
            if final_data:
                f.write(f"- Query: {text}\n")
                f.write(f"- Detected Lang Param: {final_data.get('language')}\n")
                f.write(f"- Answer: {repr(final_data.get('answer'))}\n\n")
            else:
                f.write("- Failed to get final response.\n\n")
            
    except Exception as e:
        with codecs.open(out_file, "a", "utf-8") as f:
            f.write(f"### {lang}\n")
            f.write(f"- FAILED: {e}\n\n")

if __name__ == "__main__":
    out_file = "C:\\Users\\Lenovo\\.gemini\\antigravity-ide\\brain\\bac84837-44c3-4edc-9b11-b000ed784ce7\\fallback_test_results.md"
    with codecs.open(out_file, "w", "utf-8") as f:
        f.write("# Fallback Test Results\n\n")
        
    for lang, text in queries:
        test_query(lang, text, out_file)
