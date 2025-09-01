import requests

payload = {
    "messages": [
        {"role": "system", "content": "This is a task you must complete by returning only the output.\nDo not include explanations, code, or extra text—only the result."},
        {"role": "user", "content": "Hi I am Mihiran and 24 years old from sri lanka"}
    ],
    "json_schema": {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        },
        "required": ["name", "age"],
        "additionalProperties": False
    },
    "temperature": 0.0,
    "max_tokens": 128
}

url = "https://5wwzgp9vqbnlz3-7000.proxy.runpod.net/chat"

r = requests.post(url, json=payload)
response = r.json()

print(response.get("response"))
