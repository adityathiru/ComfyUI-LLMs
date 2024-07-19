import unittest
from loguru import logger
from unittest.mock import patch, MagicMock
from llm import LLM
from llm import BaseOpenAI
from llm import BaseAnthropic
from llm.custom_typing import Conversation, UserMessage

class TestLLM(unittest.TestCase):

    def setUp(self):
        self.mock_conversation = Conversation(messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [{"type": "text", "text": "Hello!"}]}
        ])

    def test_llm_conversation_flow(self):
        conversation = Conversation(messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [{"type": "text", "text": "Hello!"}]}
        ])

        llm = LLM("openai",
                  model="gpt-4o",
                  model_params={"max_tokens": 100},
                  conversation=conversation,
                  stateful=True)()
        
        # Initial greeting
        llm.run()
        self.assertEqual(llm.conversation.messages[-2].role, "user")
        self.assertEqual(llm.conversation.messages[-2].content[0].text, "Hello!")
        self.assertEqual(llm.conversation.messages[-1].role, "assistant")

        # Climate question
        climate_question = "What is the climate in Chennai city like?"
        llm.add_message_to_conversation(UserMessage(content=[{"type": "text", "text": climate_question}]))
        llm.run()
        self.assertEqual(llm.conversation.messages[-2].role, "user")
        self.assertEqual(llm.conversation.messages[-2].content[0].text, climate_question)
        self.assertEqual(llm.conversation.messages[-1].role, "assistant")

        # Best time to visit
        visit_question = "What is the best time to visit Chennai?"
        llm.add_message_to_conversation(UserMessage(content=[{"type": "text", "text": visit_question}]))
        llm.run()
        self.assertEqual(llm.conversation.messages[-2].role, "user")
        self.assertEqual(llm.conversation.messages[-2].content[0].text, visit_question)
        self.assertEqual(llm.conversation.messages[-1].role, "assistant")

        # Itinerary request
        itinerary_request = "Write an exhaustive 10-day itinerary for Chennai and nearby places in India?"
        llm.add_message_to_conversation(UserMessage(content=[{"type": "text", "text": itinerary_request}]))
        llm.run(until_completion=True)
        self.assertEqual(llm.conversation.messages[-2].role, "user")
        self.assertEqual(llm.conversation.messages[-2].content[0].text, itinerary_request)
        self.assertEqual(llm.conversation.messages[-1].role, "assistant")

    def test_llm_initialization(self):
        for vendor, model_class in [(BaseOpenAI.VENDOR, BaseOpenAI), (BaseAnthropic.VENDOR, BaseAnthropic)]:
            with self.subTest(vendor=vendor):
                llm = LLM(vendor=vendor, model=model_class.ALLOWED_MODELS[0], 
                          model_params={}, conversation=self.mock_conversation)
                instance = llm()
                self.assertIsInstance(instance, model_class)

    def test_llm_run(self):
        for vendor, model_class in [(BaseOpenAI.VENDOR, BaseOpenAI), (BaseAnthropic.VENDOR, BaseAnthropic)]:
            with self.subTest(vendor=vendor):
                conversation = Conversation(messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": [{"type": "text", "text": "Hello!"}]}
                ])
                
                llm = LLM(vendor=vendor, model=model_class.ALLOWED_MODELS[0], 
                          model_params={"max_tokens": 2}, conversation=conversation, stateful=False)()
                
                # Test run without completion
                llm.run(until_completion=False)
                self.assertEqual(llm.conversation.messages[-1].role, "user")
                self.assertIsNotNone(llm.conversation.messages[-1].content)
                
                # Test run with completion
                llm = LLM(vendor=vendor, model=model_class.ALLOWED_MODELS[0], 
                          model_params={"max_tokens": 2}, conversation=conversation, stateful=True)()
                llm.run(until_completion=True)
                self.assertEqual(llm.conversation.messages[-1].role, "assistant")
                self.assertIsNotNone(llm.conversation.messages[-1].content)
                self.assertGreater(len(llm.conversation.messages[-1].content[0].text), 2)

    def test_llm_unknown_vendor(self):
        with self.assertRaisesRegex(ValueError, "Unknown vendor: unknown"):
            LLM(vendor="unknown", model="unknown")()

    def test_llm_invalid_model(self):
        for vendor, model_class in [(BaseOpenAI.VENDOR, BaseOpenAI), (BaseAnthropic.VENDOR, BaseAnthropic)]:
            with self.subTest(vendor=vendor):
                with self.assertRaisesRegex(ValueError, "Model invalid_model is not supported"):
                    LLM(vendor=vendor, model="invalid_model", 
                        model_params={}, conversation=self.mock_conversation)()

if __name__ == '__main__':
    unittest.main()