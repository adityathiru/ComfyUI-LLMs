import os

from .constants import *
from .nodes.predict import *
from .nodes.text_field import *
from .nodes.prompt_builder import *
from .nodes.model import *

NODE_CLASS_MAPPINGS = {
    f"{PREFIX} Text Field": TextField,
    f"{PREFIX} Prompt Builder": PromptBuilder,
    f"{PREFIX} Predict": Predict,
    f"{PREFIX} Model": Model,
}

print("\033[34mComfyUI LLM Nodes: \033[92mLoaded\033[0m")
