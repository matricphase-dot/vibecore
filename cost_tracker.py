GPT_COST_PER_TOKEN = 0.002
LOCAL_COST_PER_TOKEN = 0.0

def estimate_tokens(text: str) -> int:
    return max(1, len(text.split()))

def calculate_cost(prompt: str, source: str) -> dict:
    tokens = estimate_tokens(prompt)
    original_cost = round(tokens * GPT_COST_PER_TOKEN, 4)

    if source in ['exact_cache', 'semantic_cache']:
        optimized_cost = 0.0
    elif source == 'ollama':
        optimized_cost = 0.0
    else:
        optimized_cost = original_cost

    saved = round(original_cost - optimized_cost, 4)

    print(f'[Cost] Tokens: {tokens} | Original: Rs.{original_cost} | Saved: Rs.{saved}')

    return {
        'tokens': tokens,
        'cost_original': original_cost,
        'cost_optimized': optimized_cost,
        'saved': saved
    }
