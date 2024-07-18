
class TextField:     
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {       
                    "text": ("STRING", {"multiline": True, "default": ""}),
                    }
                }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "text_input"
    OUTPUT_NODE = True
    CATEGORY = "LLMNodes"

    def text_input(self, text):
        print("got input in text_field.py", text)
        return (text,)
