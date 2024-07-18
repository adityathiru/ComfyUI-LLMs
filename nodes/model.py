class Model:     
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vendor": ("STRING", {"default": "openai"}),
                "model": ("STRING", {"default": "gpt-4o"}),
                "max_tokens": ("INT", {"default": 4000, "min": 1}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "set_params"
    OUTPUT_NODE = True
    CATEGORY = "🤖 LLM"

    def set_params(self, vendor, model, max_tokens):
        # TODO: introduce a stateful LLM object notion
        # TODO: share the LLM object around, to maintain state for continuation, agentic workflows
        return ({"vendor": vendor, "model": model, "max_tokens": max_tokens},)
