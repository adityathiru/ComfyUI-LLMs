import os

from .constants import *
from .nodes.predict import *
from .nodes.text_field import *
from .nodes.prompt_builder import *
from .nodes.model import *

NODE_CLASS_MAPPINGS = {
    f"Text Field": TextField,
    f"Prompt Builder": PromptBuilder,
    f"Model": Model,
    f"Predict": Predict,
    f"Model V2": ModelV2,
    f"Predict V2": PredictV2,
}

print("\033[34mComfyUI LLM Nodes: \033[92mLoaded\033[0m")
