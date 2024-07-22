import os
from typing import Dict
from openai import OpenAI
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential
from .constants import SUPPORTED_MODELS
from .custom_typing import Conversation, UserMessage, AssistantMessage
from .base_llm import BaseLLM

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key=OPENAI_API_KEY)

class BaseOpenAI(BaseLLM):
    VENDOR = "openai"
    ALLOWED_MODELS = SUPPORTED_MODELS[VENDOR]

    def __init__(self, model: str, model_params: Dict[str, str] = {}, conversation: Conversation = Conversation(messages=[]), stateful: bool = True):
        logger.info(f"Initializing OpenAI with model: {model}, stateful: {stateful}")
        if model not in self.ALLOWED_MODELS:
            raise ValueError(f"Model {model} is not supported")
        super().__init__(self.VENDOR, model, model_params, conversation, stateful)

    def __convert_conversation_to_messages(self, conversation: Conversation):
        logger.debug("Converting conversation to OpenAI message format")
        """
        convert this format of conversation:
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
        
        into this format:
        messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    *[{"type": "image_url", "image_url": {"url": img_url}} for img_url in images_base64],
                    {"type": "text", "text": user_prompt},
                ]}
            ]
        """
        final_messages = []
        for message in conversation.messages:
            if message.role == "system":
                final_messages.append({"role": "system", "content": message.content})
            elif message.role == "user":
                user_content = []
                for content_item in message.content:
                    if content_item.type == "text":
                        user_content.append({"type": "text", "text": content_item.text})
                    elif content_item.type == "image":
                        user_content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{content_item.source.media_type};{content_item.source.type},{content_item.source.data}"
                            }
                        })
                final_messages.append({"role": "user", "content": user_content})
            elif message.role == "assistant":
                final_messages.append({"role": "assistant", "content": message.content})
        logger.debug(f"Converted {len(conversation.messages)} messages to OpenAI format")
        return final_messages

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _run_messages(self):
        logger.info("Running messages through OpenAI API")
        messages = self.__convert_conversation_to_messages(self.conversation)
        try:
            response = openai_client.chat.completions.create(
                model=self.model,
                messages=messages,
                **self.model_params
            )
            logger.debug("Successfully received response from OpenAI API")
            output_text = response.choices[0].message.content

            if self.stateful:
                assistant_message = AssistantMessage(content=[{"type": "text", "text": output_text}], finish_reason=response.choices[0].finish_reason)
                self.add_message_to_conversation(assistant_message)
                logger.debug("Added assistant message to conversation")
            return output_text, response
        except Exception as e:
            logger.error(f"Error occurred while running messages: {str(e)}")
            raise

    def _run_messages_until_completion(self, until_completion_user_message: UserMessage):
        logger.info("Running messages until completion")
        assert self.stateful, "stateful must be True to run until completion"

        full_output_text = []
        while True:
            logger.debug("Running model iteration")
            output_text, response = self._run_messages()
            full_output_text.append(output_text)
            finish_reason = response.choices[0].finish_reason
            logger.debug(f"Response: {response}")
            
            if finish_reason in ["stop_sequence", "stop"]:
                logger.info("Model generated a stop sequence")
                return "".join(full_output_text)
            elif finish_reason == "length":
                self.add_message_to_conversation(until_completion_user_message)
                logger.warning("Incomplete model output due to max_tokens parameter or token limit")
                continue
            elif finish_reason == "content_filter":
                logger.error("Model generated content that violates the content policy")
                raise RuntimeError("Content filter violation")
            elif finish_reason == "function_call":
                logger.error("Model called a function, which is not supported yet")
                raise NotImplementedError("Function calling is not supported yet")
            elif finish_reason in ["null", None]:
                logger.error("Model did not generate any content")
                raise RuntimeError("Model did not generate any content")
            else:
                logger.warning(f"Unknown finish reason: {finish_reason}")
                return "".join(full_output_text)

    def run(
        self,
        until_completion: bool = False,
        until_completion_user_message: UserMessage = None,
        cleanup_completion: bool = True
    ):
        logger.info(f"Running OpenAI with until_completion: {until_completion}")
        if until_completion:
            if not until_completion_user_message:
                until_completion_user_message = self.default_until_completion_user_message
            output_text = self._run_messages_until_completion(until_completion_user_message)
            if cleanup_completion:
                self.cleanup_completion(output_text, until_completion_user_message)
        else:
            output_text, response = self._run_messages()

        logger.info("Completed running OpenAI")
        return output_text
