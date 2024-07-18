import os

from .constants import *
from .nodes.predict import *
from .nodes.text_field import *
from .nodes.prompt_builder import *
from .nodes.model import *

NODE_CLASS_MAPPINGS = {
    f"Text Field": TextField,
    f"Prompt Builder": PromptBuilder,
    f"Predict": Predict,
    f"Model": Model,
}

print("\033[34mComfyUI LLM Nodes: \033[92mLoaded\033[0m")
