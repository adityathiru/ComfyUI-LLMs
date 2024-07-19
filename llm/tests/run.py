from llm.custom_typing import Conversation, UserMessage
from llm import LLM

if __name__ == "__main__":
    
    conversation = Conversation(messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [{"type": "text", "text": "Hello!"}]}
    ])

    llm = LLM("anthropic", model="claude-3-5-sonnet-20240620", conversation=conversation, stateful=True)()
    llm.run()
    print("RESPONSE", llm.get_latest_assistant_message(text=True))
    llm.add_message_to_conversation(UserMessage(content=[{"type": "text", "text": "What is the climate in Chennai city like?"}]))
    llm.run()
    print("RESPONSE", llm.get_latest_assistant_message(text=True))
    llm.add_message_to_conversation(UserMessage(content=[{"type": "text", "text": "What is the best time to visit Chennai ?"}]))
    llm.run()
    print("RESPONSE", llm.get_latest_assistant_message(text=True))
    llm.add_message_to_conversation(UserMessage(content=[{"type": "text", "text": "Write an exhaustive 10-day itinerary for Chennai and nearby places in India ?"}]))
    llm.run(until_completion=True)
    print("RESPONSE", llm.get_latest_assistant_message(text=True))
