import requests

res = requests.post(
    "http://localhost:11434/api/chat",
    json={
        "model": "llama2",
        "messages": [{"role": "user", "content": "Hello!"}]
    }
)

print(res.status_code)
print(res.text)
