from typing import List, Union, Literal
from pydantic import BaseModel, Field

class ImageSource(BaseModel):
    type: Literal["base64"] = "base64"
    media_type: str = Field(..., pattern="^image/(jpeg|png)$")
    data: str

class ImageContent(BaseModel):
    type: Literal["image"] = "image"
    source: ImageSource

class TextContent(BaseModel):
    type: Literal["text"] = "text"
    text: str

class SystemMessage(BaseModel):
    role: Literal["system"] = "system"
    content: str

class UserMessage(BaseModel):
    role: Literal["user"] = "user"
    content: List[Union[TextContent, ImageContent]]

class AssistantMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: List[Union[TextContent, ImageContent]]
    finish_reason: str


Message = Union[SystemMessage, UserMessage, AssistantMessage]

class Conversation(BaseModel):
    """
    data structure for a conversation:
    [
        {
            "role": "system",
            "content": "<str>"
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "<str>"
                },
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": "<str>"
                    }
                }
            ]
        },
        {"role": "assistant", "content": "<str>"},
    ]
    """
    messages: List[Message]
