import re

MAX_TOKENS = 100

def optimize_prompt(prompt: str) -> dict:
    original = prompt
    original_tokens = len(prompt.split())

    prompt = prompt.strip()
    prompt = re.sub(r'\s+', ' ', prompt)
    prompt = re.sub(r'(\b\w+\b)( \1\b)+', r'\1', prompt, flags=re.IGNORECASE)

    words = prompt.split()
    if len(words) > MAX_TOKENS:
        words = words[:MAX_TOKENS]
        prompt = ' '.join(words)

    optimized_tokens = len(prompt.split())
    tokens_saved = original_tokens - optimized_tokens

    print(f'[Optimizer] {original_tokens} tokens -> {optimized_tokens} tokens (saved {tokens_saved})')

    return {
        'original': original,
        'optimized': prompt,
        'original_tokens': original_tokens,
        'optimized_tokens': optimized_tokens,
        'tokens_saved': tokens_saved
    }
