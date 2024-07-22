from jinja2 import Template


class PromptBuilder:     
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {       
                "prompt_template": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "input_1": ("*", {"default": ""}),
                "input_2": ("*", {"default": ""}),
                "input_3": ("*", {"default": ""}),
                "input_4": ("*", {"default": ""}),
                "input_5": ("*", {"default": ""}),
                # Add more inputs as needed
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "process_template"
    OUTPUT_NODE = True
    CATEGORY = "🤖 LLM"

    def process_template(self, prompt_template, **kwargs):
        jinja_template = Template(prompt_template)
        for key, value in kwargs.items():
            if not value:
                # pop the key from kwargs
                kwargs.pop(key)
        result = jinja_template.render(**kwargs)

        return (result,)

    