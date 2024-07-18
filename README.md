# ComfyUI LLMs
> Goal: To enable folks to rapidly build complex workflows with LLMs
☠️ This is experimental and not recommended to use in a production environment (yet!)

## Features
1. Supports OpenAI, Anthropic models
2. Supports Image inputs for Multi-Modal LLMs
3. Supports Jinja Templates with Input Variables for building complex prompting

## Installation
1. Clone ComfyUI: `git clone https://github.com/comfyanonymous/ComfyUI.git`
2. Follow the ComfyUI setup instructions
3. Go to `custom_nodes` directory
4. Clone ComfyUI-LLMs inside ComfyUI/custom_nodes: `git clone https://github.com/adityathiru/ComfyUI-LLMs.git`
5. Install the requirements: `pip install -r requirements.txt`
6. Go back to the root of ComfyUI ; Start the ComfyUI server: `python main.py`
7. Go to the ComfyUI interface and add a new flow
8. Right-click anywhere in the flow and select "Add Node"
9. Select "LLM" from the list of options to find the LLM nodes

## Examples
### Nodes Available
![Nodes Available](examples/examples-nodes-available.png)

### Multi-Step Workflows with LLMs and PDFs
![Multi-Step Workflows with LLMs and Documents](examples/example-1.png)

## Roadmap
1. [ ] Increased LLM Support and Validations
2. [ ] Stateful LLMs: For maintaing message history
3. [ ] Agentic LLMs: ReACT-like architecture nodes
4. [ ] ComfyUI + Airflow for Job Execution
5. [ ] Customized ComfyUI for LLMs

## Appendix
### Other Useful ComfyUI Nodes for working with LLMs
1. [ComfyUI-Documents](https://github.com/Excidos/ComfyUI-Documents.git): For loading PDFs, converting it into images, and injecting them into ComfyUI-LLMs
2. [ComfyUI-Crystools](https://github.com/crystian/ComfyUI-Crystools): For various utilities like Displaying Text, Displaying Images, String Manipulation etc.
