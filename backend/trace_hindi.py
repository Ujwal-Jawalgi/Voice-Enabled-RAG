import asyncio
import urllib.request
import json
import codecs

url = 'http://127.0.0.1:8000/query'
headers = {
    'Content-Type': 'application/json'
}

def test_query():
    text = "गोवा की राजधानी क्या है?"
    data = json.dumps({'text': text}).encode('utf-8')
    req = urllib.request.Request(url, data=data, headers=headers)
    
    response = urllib.request.urlopen(req)
    body = response.read().decode('utf-8')
    
    with codecs.open("live_hindi_trace.md", "w", "utf-8") as f:
        f.write(body)

if __name__ == "__main__":
    test_query()
