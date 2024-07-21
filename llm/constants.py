SUPPORTED_MODELS = {
    "openai": [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4-vision-preview"
    ],
    "anthropic": [
        "claude-3-5-sonnet-20240620",
        "claude-3-opus-20240229",
        "claude-3-haiku-20240307"
    ]
}

def flat_vendor_models():
    """
    Returns a list of all models supported by the LLM nodes, in the format "vendor/model".
    """
    return [f"{vendor}/{model}" for vendor, models in SUPPORTED_MODELS.items() for model in models]
