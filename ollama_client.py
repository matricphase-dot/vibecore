import requests

OLLAMA_URL = 'http://localhost:11434/api/generate'
MODEL = 'tinyllama'

def generate_local(prompt: str) -> str:
    try:
        response = requests.post(OLLAMA_URL, json={
            'model': MODEL,
            'prompt': prompt,
            'stream': False
        })
        data = response.json()
        print(f'[Ollama] Generated response successfully')
        return data.get('response', '').strip()
    except Exception as e:
        print(f'[Ollama] Error: {e}')
        return None
