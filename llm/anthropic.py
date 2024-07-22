import os
from typing import Dict
from copy import deepcopy
from anthropic import Anthropic
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential
from .constants import SUPPORTED_MODELS
from .custom_typing import Conversation, UserMessage, AssistantMessage
from .base_llm import BaseLLM

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)

class BaseAnthropic(BaseLLM):
    VENDOR = "anthropic"
    ALLOWED_MODELS = SUPPORTED_MODELS[VENDOR]

    def __init__(self, model: str, model_params: Dict[str, str] = {}, conversation: Conversation = Conversation(messages=[]), stateful: bool = True):
        logger.info(f"Initializing Anthropic with model: {model}, stateful: {stateful}")
        if model not in self.ALLOWED_MODELS:
            raise ValueError(f"Model {model} is not supported")
        super().__init__(self.VENDOR, model, model_params, conversation, stateful)

    def __convert_conversation_to_messages(self, conversation: Conversation):
        logger.debug("Converting conversation to Anthropic message format")
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
                            "type": "image",
                            "source": {
                                "type": content_item.source.type,
                                "media_type": content_item.source.media_type,
                                "data": content_item.source.data
                            }
                        })
                final_messages.append({"role": "user", "content": user_content})
            elif message.role == "assistant":
                final_messages.append({"role": "assistant", "content": message.content})
        logger.debug(f"Converted {len(conversation.messages)} messages to Anthropic format")
        return final_messages

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _run_messages(self):
        logger.info("Running messages through OpenAI API")
        messages = self.__convert_conversation_to_messages(self.conversation)
        messages = deepcopy(messages)
        try:
            # pop the first system message out and store the system message in a text variable "system" and pass to anthropic client
            system_message = messages.pop(0)
            system_text = system_message["content"]
            if "max_tokens" not in self.model_params:
                self.model_params["max_tokens"] = 4000
            response = anthropic_client.messages.create(
                model=self.model,
                system=system_text,
                messages=messages,
                **self.model_params
            )
            logger.debug("Successfully received response from Anthropic API")
            output_text = response.content[0].text

            if self.stateful:
                assistant_message = AssistantMessage(content=[{"type": "text", "text": output_text}], finish_reason=response.stop_reason)
                self.add_message_to_conversation(assistant_message)
                logger.debug("Added assistant message to conversation")
            return output_text, response
        except Exception as e:
            logger.error(f"Error occurred while running messages: {str(e)}")
            raise

    def _run_messages_until_completion(self, until_completion_user_message: UserMessage):
        logger.info("Running messages until completion")
        assert self.stateful, "stateful must be True to run until completion"
        
        while True:
            logger.debug("Running model iteration")
            _, response = self._run_messages()
            logger.debug(f"Response: {response}")
            finish_reason = response.stop_reason
            
            if finish_reason in ["end_turn", "stop_sequence"]:
                logger.info("Model generated a stop sequence")
                return
            elif finish_reason == "max_tokens":
                logger.warning("Incomplete model output due to max_tokens parameter or token limit")
                continue
            elif finish_reason == "tool_use":
                logger.error("Model called a function, which is not supported yet")
                raise NotImplementedError("Function calling is not supported yet")
            elif finish_reason in ["null", None]:
                logger.error("Model did not generate any content")
                raise RuntimeError("Model did not generate any content")
            else:
                logger.warning(f"Unknown finish reason: {finish_reason}")
                return

    def run(
        self,
        until_completion: bool = False,
        until_completion_user_message: UserMessage = None,
        cleanup_completion: bool = True
    ):
        logger.info(f"Running Anthropic with until_completion: {until_completion}")
        if until_completion:
            if not until_completion_user_message:
                until_completion_user_message = self.default_until_completion_user_message
            output_text = self._run_messages_until_completion(until_completion_user_message)
            if cleanup_completion:
                self.cleanup_completion(output_text, until_completion_user_message)
        else:
            output_text, response = self._run_messages()
        logger.info("Completed running Anthropic")
        return output_text
