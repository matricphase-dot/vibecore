COMPLEX_KEYWORDS = [
    'analyze', 'analysis', 'strategy', 'complex', 'compare',
    'explain in detail', 'research', 'summarize', 'evaluate',
    'pros and cons', 'difference between', 'how does', 'why does'
]

def classify_prompt(prompt: str) -> str:
    prompt_lower = prompt.lower()
    word_count = len(prompt.split())

    for keyword in COMPLEX_KEYWORDS:
        if keyword in prompt_lower:
            print(f'[Classifier] COMPLEX - matched keyword: {keyword}')
            return 'complex'

    if word_count > 20:
        print(f'[Classifier] COMPLEX - too long: {word_count} words')
        return 'complex'

    print(f'[Classifier] SIMPLE - {word_count} words, no complex keywords')
    return 'simple'
