@app.get('/debug')
async def debug():
    try:
        r = get_redis()
        r.ping()
        return {'redis': 'connected', 'url_prefix': REDIS_URL[:15]}
    except Exception as e:
        return {'redis': 'failed', 'error': str(e), 'url_prefix': REDIS_URL[:15]}

import os
# import startup
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from datetime import datetime
import hashlib, time, requests as req

REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379')
GROQ_API_KEY = os.getenv('GROQ_API_KEY', '')

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])

total_saved = 0.0
total_requests = 0
total_response_ms = 0
sources = {'groq': 0, 'exact_cache': 0, 'semantic_cache': 0, 'external_api': 0}
recent_requests = []

_redis_client = None

def get_redis():
    global _redis_client
    if _redis_client is None:
        import redis as redis_lib
        _redis_client = redis_lib.from_url(REDIS_URL, decode_responses=True)
    return _redis_client

class PromptRequest(BaseModel):
    prompt: str

class SignupRequest(BaseModel):
    email: str

def hash_prompt(prompt: str, api_key: str) -> str:
    return hashlib.sha256(f'{api_key}:{prompt.strip().lower()}'.encode()).hexdigest()

def generate_groq(prompt: str) -> str:
    if not GROQ_API_KEY:
        return f'Echo: {prompt}'
    try:
        res = req.post('https://api.groq.com/openai/v1/chat/completions',
            headers={'Authorization': f'Bearer {GROQ_API_KEY}', 'Content-Type': 'application/json'},
            json={'model': 'llama-3.3-70b-versatile', 'messages': [{'role': 'user', 'content': prompt}], 'max_tokens': 500})
        return res.json()['choices'][0]['message']['content'].strip()
    except Exception as e:
        print(f'[Groq] Error: {e}')
        return f'Echo: {prompt}'

def create_user(email: str) -> dict:
    import json, uuid
    cache = get_redis()
    existing = cache.get(f'email:{email}')
    if existing:
        return json.loads(existing)
    api_key = 'vc_live_' + uuid.uuid4().hex[:24]
    user = {'email': email, 'api_key': api_key, 'created_at': str(datetime.now()), 'total_requests': 0, 'total_saved': 0.0, 'plan': 'free', 'limit': 1000}
    cache.set(f'user:{api_key}', json.dumps(user))
    cache.set(f'email:{email}', json.dumps(user))
    return user

def get_user(api_key: str) -> dict:
    import json
    cache = get_redis()
    data = cache.get(f'user:{api_key}')
    if not data:
        return None
    return json.loads(data)

def update_user_stats(api_key: str, saved: float):
    import json
    cache = get_redis()
    user = get_user(api_key)
    if not user:
        return
    user['total_requests'] += 1
    user['total_saved'] = round(user['total_saved'] + saved, 4)
    cache.set(f'user:{api_key}', json.dumps(user))

@app.get('/')
async def landing():
    return FileResponse('index.html')

@app.get('/dashboard')
async def dashboard():
    return FileResponse('dashboard.html')

@app.get('/health')
async def health():
    return {'status': 'ok'}

@app.post('/signup')
async def signup(request: SignupRequest):
    try:
        user = create_user(request.email)
        return {'message': 'Welcome to VibeCore!', 'email': user['email'], 'api_key': user['api_key'], 'plan': user['plan'], 'limit': user['limit']}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/me')
async def me(x_api_key: str = Header(...)):
    user = get_user(x_api_key)
    if not user:
        raise HTTPException(status_code=401, detail='Invalid API key')
    return {'email': user['email'], 'api_key': user['api_key'], 'plan': user['plan'], 'total_requests': user['total_requests'], 'total_saved': user['total_saved'], 'limit': user['limit'], 'requests_remaining': user['limit'] - user['total_requests']}

@app.get('/stats')
async def stats(x_api_key: str = Header(None)):
    if x_api_key:
        user = get_user(x_api_key)
        if user:
            return {'total_saved': user['total_saved'], 'total_requests': user['total_requests'], 'requests_remaining': user['limit'] - user['total_requests'], 'plan': user['plan']}
    hit_rate = 0
    if total_requests > 0:
        hits = sources.get('exact_cache', 0) + sources.get('semantic_cache', 0)
        hit_rate = round((hits / total_requests) * 100, 1)
    avg_ms = round(total_response_ms / total_requests) if total_requests > 0 else 0
    return {'total_saved': round(total_saved, 4), 'total_requests': total_requests, 'hit_rate': hit_rate, 'avg_response_ms': avg_ms, 'sources': sources, 'recent_requests': recent_requests[-10:]}

@app.post('/generate')
async def generate(request: PromptRequest, x_api_key: str = Header(...)):
    global total_saved, total_requests, total_response_ms
    start = time.time()

    user = get_user(x_api_key)
    if not user:
        raise HTTPException(status_code=401, detail='Invalid API key. Get one at https://vibecore-07n6.onrender.com')

    if user['total_requests'] >= user['limit']:
        raise HTTPException(status_code=429, detail=f'Free limit reached. Upgrade at vibecore-07n6.onrender.com')

    print(f'\n[{datetime.now()}] User: {user["email"]} | Prompt: {request.prompt[:40]}')

    from optimizer import optimize_prompt
    from classifier import classify_prompt
    from cost_tracker import calculate_cost

    optimized = optimize_prompt(request.prompt)
    prompt = optimized['optimized']
    cache = get_redis()
    key = hash_prompt(prompt, x_api_key)

    exact = cache.get(key)
    if exact:
        cost = calculate_cost(prompt, 'exact_cache')
        total_saved += cost['saved']
        total_requests += 1
        ms = round((time.time() - start) * 1000)
        total_response_ms += ms
        sources['exact_cache'] += 1
        recent_requests.append({'prompt': prompt[:50], 'source': 'exact_cache', 'saved': cost['saved'], 'response_ms': ms})
        update_user_stats(x_api_key, cost['saved'])
        return {'response': exact, 'cached': True, 'source': 'exact_cache', 'tokens': cost['tokens'], 'cost_original': cost['cost_original'], 'cost_optimized': cost['cost_optimized'], 'saved': cost['saved'], 'total_saved': round(total_saved, 4)}

    complexity = classify_prompt(prompt)
    response_text = generate_groq(prompt)
    source = 'groq'

    cache.setex(key, 3600, response_text)
    cost = calculate_cost(prompt, source)
    total_saved += cost['saved']
    total_requests += 1
    ms = round((time.time() - start) * 1000)
    total_response_ms += ms
    sources[source] += 1
    recent_requests.append({'prompt': prompt[:50], 'source': source, 'saved': cost['saved'], 'response_ms': ms})
    update_user_stats(x_api_key, cost['saved'])

    return {'response': response_text, 'cached': False, 'source': source, 'complexity': complexity, 'tokens': cost['tokens'], 'cost_original': cost['cost_original'], 'cost_optimized': cost['cost_optimized'], 'saved': cost['saved'], 'total_saved': round(total_saved, 4)}
