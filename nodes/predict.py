import os
import openai
from anthropic import Anthropic
from openai import OpenAI
from PIL import Image
import numpy as np
import base64
import io


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

openai_client = OpenAI(api_key=OPENAI_API_KEY)
anthropic_client = anthropic = Anthropic(api_key=ANTHROPIC_API_KEY)


class Predict:
    @classmethod
    def INPUT_TYPES(cls):
        return {
			"required": {
                    "system_prompt": ("STRING", {"multiline": False, "forceInput": True}),
                    "user_prompt": ("STRING", {"multiline": False, "forceInput": True}),
                    "model_details": ("MODEL", {"forceInput": True}),
			},
            "optional": {
                "images": ("IMAGE", {"multiple": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "predict"
    OUTPUT_NODE = True
    CATEGORY = "🤖 LLM"

    def images_to_base64(self, images):
        images_base64 = []
        for (batch_number, image) in enumerate(images):
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            
            # Resize the image to a maximum width or height of 1024 pixels
            max_size = 1024
            img.thumbnail((max_size, max_size), Image.LANCZOS)
            
            # Convert to RGB mode if the image is in RGBA mode
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            
            # Save as JPEG with reduced quality
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=85)
            img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            images_base64.append(f"data:image/jpeg;base64,{img_base64}")
        return images_base64

    def predict(self, system_prompt, user_prompt, model_details, images=[]):
        vendor = model_details['vendor']
        model = model_details['model']
        max_tokens = model_details['max_tokens']
        if len(images) > 0:
            images_base64 = self.images_to_base64(images)
        else:
            images_base64 = []

        if vendor.lower() == 'openai':
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    *[{"type": "image_url", "image_url": {"url": img_url}} for img_url in images_base64],
                    {"type": "text", "text": user_prompt},
                ]}
            ]
            response = openai_client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens
            )
            output_text = response.choices[0].message.content
        elif vendor.lower() == 'anthropic':
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    *[{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": img_url.split(",")[1]}} for img_url in images_base64],
                    {"type": "text", "text": user_prompt},
                ]}
            ]
            response = anthropic_client.messages.create(
                model=model,
                max_tokens=max_tokens,
                messages=messages
            )
            output_text = response.content[0].text
        else:
            raise ValueError(f"Unsupported vendor: {vendor}")

        return (output_text,)

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return lib0246.check_update(kwargs["_query"])
