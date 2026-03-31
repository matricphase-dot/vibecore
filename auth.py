import os
import uuid
import redis
import json
from datetime import datetime

REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379')
cache = redis.from_url(REDIS_URL, decode_responses=True)

def generate_api_key() -> str:
    return 'vc_live_' + uuid.uuid4().hex[:24]

def create_user(email: str) -> dict:
    existing = cache.get(f'email:{email}')
    if existing:
        return json.loads(existing)

    api_key = generate_api_key()
    user = {
        'email': email,
        'api_key': api_key,
        'created_at': str(datetime.now()),
        'total_requests': 0,
        'total_saved': 0.0,
        'plan': 'free',
        'limit': 1000
    }
    cache.set(f'user:{api_key}', json.dumps(user))
    cache.set(f'email:{email}', json.dumps(user))
    print(f'[Auth] Created user: {email} with key: {api_key}')
    return user

def get_user(api_key: str) -> dict:
    data = cache.get(f'user:{api_key}')
    if not data:
        return None
    return json.loads(data)

def update_user_stats(api_key: str, saved: float):
    user = get_user(api_key)
    if not user:
        return
    user['total_requests'] += 1
    user['total_saved'] = round(user['total_saved'] + saved, 4)
    cache.set(f'user:{api_key}', json.dumps(user))
