from ..llm import LLM, Conversation
from ..llm.constants import flat_vendor_models

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

class ModelV2:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (flat_vendor_models(), {"default": "openai/gpt-4o"}),
                "stateful": ("BOOLEAN", {"default": False}),
                "max_tokens": ("INT", {"default": 4000, "min": 1}),
                "temperature": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0}),
            },
            # "optional": {
            #     "complete_if_out_of_tokens": ("BOOLEAN", {"default": True}),
            #     "cleanup_out_of_token_completion": ("BOOLEAN", {"default": True}),
            # }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "set_params"
    OUTPUT_NODE = True
    CATEGORY = "🤖 LLM"

    def set_params(self, model_name, stateful, max_tokens, temperature):
        model_params = {"max_tokens": max_tokens, "temperature": temperature}
        vendor, model_name = model_name.split("/")
        llm = LLM(vendor, model_name, model_params, stateful=stateful)()
        return (llm,)

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return lib0246.check_update(kwargs["_query"])
