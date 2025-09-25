import openai
from config.config import get_config
from typing import List

VOWELS = {'A', 'E', 'I', 'O', 'U'}


def g_cupl(category: str, n_responses: int = 10, max_tokens: int = 50, temperature: float = 0.99) -> List[str]:
    
    config = get_config()
    openai.api_key = config['openai']['api_key']

    # Determine article
    article = "an" if category[0].upper() in VOWELS else "a"

    prompts = [
        f"Describe what {article} {category} looks like",
        f"How can you identify {article} {category}?",
        f"What does {article} {category} look like?",
        f"Describe an image from the internet of {article} {category}",
        f"A caption of an image of {article} {category}:"
    ]

    all_results = []

    for prompt in prompts:
        try:
            response = openai.Completion.create(
                engine="gpt-3.5-turbo-instruct",
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                n=n_responses,
                stop="."
            )
            # Process responses
            for choice in response.get("choices", []):
                text = choice.get("text", "").replace("\n", " ").replace("\"", "").strip()
                if len(text) > 10:
                    all_results.append(text + ".")
        except Exception as e:
            print(f"[ERROR] Failed to generate prompt for '{category}': {e}")

    return all_results
