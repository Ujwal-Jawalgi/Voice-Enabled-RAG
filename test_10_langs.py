import urllib.request
import json
import time
import os

url = 'https://pikchau-rag.up.railway.app/query'
headers = {
    'Content-Type': 'application/json',
    'Origin': 'https://voice-enabled-rag.vercel.app'
}

queries = [
    ("Hindi", "भारत की राजधानी क्या है?"),
    ("Bengali", "কলকাতার সেরা খাবার কি?"),
    ("Marathi", "पुण्यातील प्रसिद्ध ठिकाणे कोणती आहेत?"),
    ("Telugu", "హైదరాబాద్ లో చూడవలసిన ప్రదేశాలు ఏమిటి?"),
    ("Tamil", "இந்தியாவின் வரலாறு பற்றி சொல்லுங்கள்."),
    ("Gujarati", "ગુજરાતની રાજધાની કઈ છે?"),
    ("Urdu", "ہندوستان کی تاریخ کے بارے میں بتائیں"),
    ("Kannada", "ಬೆಂಗಳೂರಿನಲ್ಲಿ ನೋಡಬೇಕಾದ ಸ್ಥಳಗಳು ಯಾವುವು?"),
    ("Malayalam", "കേരളത്തിലെ പ്രധാന വിനോദസഞ്ചാര കേന്ദ്രങ്ങൾ ഏതാണ്?"),
    ("Punjabi", "ਪੰਜਾਬ ਦੇ ਪ੍ਰਸਿੱਧ ਤਿਉਹਾਰ ਕਿਹੜੇ ਹਨ?"),
]

output_file = "C:\\Users\\Lenovo\\.gemini\\antigravity-ide\\brain\\bac84837-44c3-4edc-9b11-b000ed784ce7\\multilingual_test_results.md"

def test_query(lang, text, out_file):
    print(f"Testing {lang}...")
    data = json.dumps({'text': text}).encode('utf-8')
    req = urllib.request.Request(url, data=data, headers=headers)
    
    start_time = time.time()
    try:
        response = urllib.request.urlopen(req)
        body = response.read().decode('utf-8')
        
        final_data = None
        try:
            final_data = json.loads(body)
        except json.JSONDecodeError:
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
            f.write(f"- **Query**: {text}\n")
            if final_data:
                f.write(f"- **Detected Language**: {final_data.get('language')}\n")
                f.write(f"- **Response**: {final_data.get('answer')}\n")
                f.write(f"- **Total Latency**: {final_data.get('timings_ms', {}).get('total', 0):.1f} ms\n")
            else:
                f.write("- **FAILED**: No final response found in SSE stream\n")
                f.write(f"- **Raw**: {body[:200]}\n")
            f.write(f"- **Roundtrip Time**: {(time.time() - start_time) * 1000:.1f} ms\n\n")
            
    except Exception as e:
        with open(out_file, "a", encoding="utf-8") as f:
            f.write(f"### {lang}\n")
            f.write(f"- **Query**: {text}\n")
            f.write(f"- **FAILED**: {e}\n\n")

if __name__ == "__main__":
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("# Multilingual Query Test Results\n\n")
    for lang, text in queries:
        test_query(lang, text, output_file)
    print("Done! Check the artifact.")
