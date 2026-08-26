import urllib.request
import json

req = urllib.request.Request("http://127.0.0.1:8000/query", method="OPTIONS")
req.add_header("Origin", "http://localhost:3000")
req.add_header("Access-Control-Request-Method", "POST")

try:
    resp = urllib.request.urlopen(req)
    print("STATUS:", resp.status)
    print("HEADERS:", resp.headers)
except Exception as e:
    print("ERROR:", e)
