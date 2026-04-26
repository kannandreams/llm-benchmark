import base64
import requests
import json
from pathlib import Path

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "gemma4:latest"
IMAGE_PATH = "test1.jpeg"

prompt = """
Describe this image in detail.
Include all visible text, tables, objects, and layout.
Minimum 500 words.
"""

image_b64 = base64.b64encode(Path(IMAGE_PATH).read_bytes()).decode("utf-8")

payload = {
    "model": MODEL,
    "prompt": prompt,
    "images": [image_b64],
    "stream": False,
    "keep_alive": "10m",
    "options": {
        "num_ctx": 32768,
        "num_predict": 1200,
        "temperature": 0.1,
        "top_k": 64,
        "top_p": 0.95,
        "seed": 42
    }
}

r = requests.post(OLLAMA_URL, json=payload)
data = r.json()

print(data["response"])

print("\n--- Metrics ---")
print("Total duration sec:", data["total_duration"] / 1e9)
print("Load duration sec:", data.get("load_duration", 0) / 1e9)
print("Prompt tokens:", data.get("prompt_eval_count"))
print("Output tokens:", data.get("eval_count"))
print("Prompt eval sec:", data.get("prompt_eval_duration", 0) / 1e9)
print("Output eval sec:", data.get("eval_duration", 0) / 1e9)

if data.get("eval_count") and data.get("eval_duration"):
    print("Tokens/sec:", data["eval_count"] / (data["eval_duration"] / 1e9))