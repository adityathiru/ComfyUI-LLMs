from typing import Any, List, Dict
from .custom_typing import Conversation, Message, UserMessage

class BaseLLM:
    def __init__(
        self,
        vendor: str,
        model: str,
        model_params: Dict[str, Any] = {},
        conversation: Conversation = Conversation(messages=[]),
        stateful: bool = True
    ):
        self.vendor = vendor
        self.model = model
        self.model_params = model_params

        self.conversation = conversation
        self.stateful = stateful

        self.default_until_completion_user_message = UserMessage(
            content=[
                {
                    "type": "text",
                    "text": "Continue. Do not add any prefix fillers that you are continuing. Continue from wherever you left of."
                }
            ]
        )

    """
    RUN
    """
    def run(self, until_completion: bool = False, until_completion_user_message: UserMessage = None):
        raise NotImplementedError("run must be implemented by subclass")

    """
    CONVERSATION HELPERS
    """
    
    def add_message_to_conversation(self, message: Message):
        self.conversation.messages.append(message)

    def get_latest_assistant_message(self, text=False):
        for message in reversed(self.conversation.messages):
            if message.role == "assistant":
                if text:
                    return message.content[0].text
                else:
                    return message
        return None

    def get_all_assistant_messages(self, text=False):
        if text:
            return [message.content[0].text for message in self.conversation.messages if message.role == "assistant"]
        else:
            return [message for message in self.conversation.messages if message.role == "assistant"]

    def cleanup_completion(self, full_output_text, until_completion_user_message: UserMessage):
        """
        there will be a bunch of assistant and user messages for continue, so we need to clean them up
        
        logic:
        1. find the last user message that's not until_completion_user_message
        2. remove all until_completion_user_message messages after that
        3. remove all assistant messages after that
        4. insert full_output_text into the conversation as an assistant message
        """
        # 1. find the last user message that's not until_completion_user_message
        last_user_message = None
        for message in self.conversation.messages:
            if message.role == "user" and message.content[0].text != until_completion_user_message.content[0].text:
                last_user_message = message

        # 2. remove all messages after the last user message that's not until_completion_user_message
        if last_user_message:
            last_user_index = self.conversation.messages.index(last_user_message)
            self.conversation.messages = self.conversation.messages[:last_user_index + 1]
        else:
            # If no other user message found, clear all messages
            logger.warning("No previous user messages found, clearing conversation")
            return

        # 4. insert full_output_text into the conversation as an assistant message
        self.conversation.messages.append(AssistantMessage(content=[{"type": "text", "text": full_output_text}], finish_reason="stop"))
        logger.info("Cleaned up completion conversation")
        return

    """
    RESPONSE PARSING
    """
    def parse_response(self, response: str, mode: str = "json", auto_fix: bool = True):
        if mode == "json":
            # try to load with ast.literal_eval first
            try:
                response_dict = ast.literal_eval(response)
            except:
                try:
                    response_dict = json.loads(response)
                except:
                    if auto_fix:
                        response_dict = self.fix_json(response)
                    else:
                        raise ValueError("Failed to parse response")
        elif mode == "text":
            return response
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def fix_json(self, json_string: str):
        raise NotImplementedError("fix_json must be implemented by subclass")
