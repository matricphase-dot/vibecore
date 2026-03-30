import os
import startup
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from datetime import datetime
from sentence_transformers import SentenceTransformer
from optimizer import optimize_prompt
from classifier import classify_prompt
from cost_tracker import calculate_cost
import hashlib, redis, numpy as np, time, requests as req

REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379')
GROQ_API_KEY = os.getenv('GROQ_API_KEY', '')
PORT = int(os.getenv('PORT', 8000))

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])

cache = redis.from_url(REDIS_URL, decode_responses=True)
model = SentenceTransformer('all-MiniLM-L6-v2')

semantic_store = []
total_saved = 0.0
total_requests = 0
total_response_ms = 0
sources = {'ollama': 0, 'exact_cache': 0, 'semantic_cache': 0, 'external_api': 0}
recent_requests = []

class PromptRequest(BaseModel):
    prompt: str

def hash_prompt(prompt: str) -> str:
    return hashlib.sha256(prompt.strip().lower().encode()).hexdigest()

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def find_semantic_match(embedding, threshold=0.90):
    best_score = 0
    best_response = None
    for entry in semantic_store:
        score = cosine_similarity(embedding, entry['embedding'])
        if score >= threshold and score > best_score:
            best_score = score
            best_response = entry['response']
    if best_response:
        print(f'[Semantic Cache] HIT with score {best_score:.4f}')
        return best_response
    return None

def generate_groq(prompt: str) -> str:
    if not GROQ_API_KEY:
        return f'Echo: {prompt}'
    try:
        res = req.post('https://api.groq.com/openai/v1/chat/completions',
            headers={'Authorization': f'Bearer {GROQ_API_KEY}', 'Content-Type': 'application/json'},
            json={'model': 'llama3-8b-8192', 'messages': [{'role': 'user', 'content': prompt}], 'max_tokens': 500})
        return res.json()['choices'][0]['message']['content'].strip()
    except Exception as e:
        print(f'[Groq] Error: {e}')
        return f'Echo: {prompt}'

@app.get('/')
async def dashboard():
    return FileResponse('dashboard.html')

@app.get('/stats')
async def stats():
    hit_rate = 0
    if total_requests > 0:
        hits = sources.get('exact_cache', 0) + sources.get('semantic_cache', 0)
        hit_rate = round((hits / total_requests) * 100, 1)
    avg_ms = round(total_response_ms / total_requests) if total_requests > 0 else 0
    return {
        'total_saved': round(total_saved, 4),
        'total_requests': total_requests,
        'hit_rate': hit_rate,
        'avg_response_ms': avg_ms,
        'sources': sources,
        'recent_requests': recent_requests[-10:],
        'message': f'You have saved Rs.{round(total_saved, 4)} so far!'
    }

@app.post('/generate')
async def generate(request: PromptRequest):
    global total_saved, total_requests, total_response_ms
    start = time.time()
    print(f'\n[{datetime.now()}] Received: {request.prompt[:60]}')

    optimized = optimize_prompt(request.prompt)
    prompt = optimized['optimized']

    key = hash_prompt(prompt)
    exact = cache.get(key)
    if exact:
        print('[Exact Cache] HIT')
        cost = calculate_cost(prompt, 'exact_cache')
        total_saved += cost['saved']
        total_requests += 1
        ms = round((time.time() - start) * 1000)
        total_response_ms += ms
        sources['exact_cache'] += 1
        recent_requests.append({'prompt': prompt[:50], 'source': 'exact_cache', 'saved': cost['saved'], 'response_ms': ms})
        return {'response': exact, 'cached': True, 'source': 'exact_cache', 'complexity': 'n/a', 'tokens': cost['tokens'], 'cost_original': cost['cost_original'], 'cost_optimized': cost['cost_optimized'], 'saved': cost['saved'], 'total_saved': round(total_saved, 4)}

    embedding = model.encode(prompt).tolist()
    semantic = find_semantic_match(embedding)
    if semantic:
        cost = calculate_cost(prompt, 'semantic_cache')
        total_saved += cost['saved']
        total_requests += 1
        ms = round((time.time() - start) * 1000)
        total_response_ms += ms
        sources['semantic_cache'] += 1
        recent_requests.append({'prompt': prompt[:50], 'source': 'semantic_cache', 'saved': cost['saved'], 'response_ms': ms})
        return {'response': semantic, 'cached': True, 'source': 'semantic_cache', 'complexity': 'n/a', 'tokens': cost['tokens'], 'cost_original': cost['cost_original'], 'cost_optimized': cost['cost_optimized'], 'saved': cost['saved'], 'total_saved': round(total_saved, 4)}

    complexity = classify_prompt(prompt)
    print(f'[Decision] Complexity: {complexity}')
    response_text = generate_groq(prompt)
    source = 'ollama' if complexity == 'simple' else 'external_api'

    cache.setex(key, 3600, response_text)
    semantic_store.append({'embedding': embedding, 'response': response_text})
    cost = calculate_cost(prompt, source)
    total_saved += cost['saved']
    total_requests += 1
    ms = round((time.time() - start) * 1000)
    total_response_ms += ms
    sources[source] += 1
    recent_requests.append({'prompt': prompt[:50], 'source': source, 'saved': cost['saved'], 'response_ms': ms})

    return {'response': response_text, 'cached': False, 'source': source, 'complexity': complexity, 'tokens': cost['tokens'], 'cost_original': cost['cost_original'], 'cost_optimized': cost['cost_optimized'], 'saved': cost['saved'], 'total_saved': round(total_saved, 4)}
