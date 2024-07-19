from .openai import BaseOpenAI
from .anthropic import BaseAnthropic
from .custom_typing import Conversation, SystemMessage, UserMessage, AssistantMessage
from typing import Dict

class LLM:
    def __init__(self, vendor: str, model: str, model_params: Dict[str, str] = {}, conversation: Conversation = Conversation(messages=[]), stateful: bool = True):
        self.vendor = vendor
        self.kwargs = {
            "model": model,
            "model_params": model_params,
            "conversation": conversation,
            "stateful": stateful
        }
    def __call__(self):
        if self.vendor == BaseOpenAI.VENDOR:
            return BaseOpenAI(**self.kwargs)
        elif self.vendor == BaseAnthropic.VENDOR:
            return BaseAnthropic(**self.kwargs)
        else:
            raise ValueError(f"Unknown vendor: {self.vendor}")
