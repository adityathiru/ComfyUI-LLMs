import os

from .nodes.predict import *
from .nodes.text_field import *
from .nodes.prompt_builder import *
from .nodes.model import *

NODE_CLASS_MAPPINGS = {
    "Text Field": TextField,
    "Prompt Builder": PromptBuilder,
    "Predict": Predict,
    "Model": Model,
}

print("\033[34mComfyUI Tutorial Nodes: \033[92mLoaded\033[0m")
