import urllib.request
import json
import time

url = 'http://127.0.0.1:8000/query'
headers = {
    'Content-Type': 'application/json'
}

queries = [
    ("English", "What is the capital of Goa?", "auto"),
    ("Hindi", "गोवा की राजधानी क्या है?", "auto"),
    ("Kannada", "ಗೋವಾದ ರಾಜಧಾನಿ ಯಾವುದು?", "auto"),
]

output_file = "C:\\Users\\Lenovo\\.gemini\\antigravity-ide\\brain\\bac84837-44c3-4edc-9b11-b000ed784ce7\\lang_test_results.md"

def test_query(lang, text, req_lang, out_file):
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
        
        with open(out_file, "a", encoding="utf-8") as f:
            f.write(f"### {lang}\n")
            if final_data:
                f.write(f"- Query: {text}\n")
                f.write(f"- Detected Lang Param: {final_data.get('language')}\n")
                f.write(f"- Answer: {final_data.get('answer')}\n\n")
            else:
                f.write("- Failed to get final response.\n\n")
            
    except Exception as e:
        with open(out_file, "a", encoding="utf-8") as f:
            f.write(f"### {lang}\n")
            f.write(f"- FAILED: {e}\n\n")

if __name__ == "__main__":
    time.sleep(5) # Give the backend a few extra seconds just in case
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("# Language Bug Fix Test\n\n")
    for lang, text, req_lang in queries:
        test_query(lang, text, req_lang, output_file)
