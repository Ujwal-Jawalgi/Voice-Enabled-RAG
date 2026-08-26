import urllib.request
import json

req = urllib.request.Request("http://127.0.0.1:8000/query", method="POST")
req.add_header("Origin", "http://localhost:3000")
req.add_header("Content-Type", "application/json")
data = json.dumps({"text": "test", "session_id": "123"}).encode('utf-8')

try:
    resp = urllib.request.urlopen(req, data=data)
    print("STATUS:", resp.status)
    print("HEADERS:", resp.headers)
    print("BODY:", resp.read().decode('utf-8'))
except Exception as e:
    print("ERROR:", e)
