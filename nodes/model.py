class Model:     
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {  
                    "vendor": ("STRING", {"default": "openai"}),
                    "model": ("STRING", {"default": "gpt-4o"}),
                    "max_tokens": ("INT", {"default": 4000, "min": 1}),
                    }
                }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "set_params"
    OUTPUT_NODE = True
    CATEGORY = "LLMNodes"

    def set_params(self, vendor, model, max_tokens):
        return ({"vendor": vendor, "model": model, "max_tokens": max_tokens},)
