import collections
import datetime
import io
import os
import openai
import dotenv

MODEL = "gpt-3.5-turbo-1106"

dotenv.load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

Message = collections.namedtuple('Message', ['role', 'content'])

class ChatGPT:
    debug: bool
    system_message: Message
    messages: list[Message]

    def __init__(self, debug: bool = False, system_message: Message = None):
        self.debug = debug
        self.client = openai.OpenAI(timeout=60)
        self.debug_log_filename = f"logs/chatgpt-{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}-debug.log"
        self.system_message = system_message
        self.messages = []

    def respond(self, message) -> str:
        self.add_user_message(message)
        response = self.client.chat.completions.create(
            model=MODEL,
            messages=self.get_messages(),
        )
        response_message = response.choices[0].message.content
        self.add_assistant_message(response_message)
        if self.debug:
            self.debug_log(self.messages_to_string())

        return response_message

    def get_messages(self):
        messages = self.messages
        if self.system_message != None:
            messages = [self.system_message] + messages
        return [m._asdict() for m in messages]

    def messages_to_string(self):
        messages = self.messages
        if self.system_message != None:
            messages = [self.system_message] + messages
        return "\n".join([f"{m.role}: {m.content}" for m in messages])

    def add_message(self, role: str, content: str):
        self.messages.append(Message(role, content))

    def add_user_message(self, content: str):
        self.add_message("user", content)

    def add_assistant_message(self, content: str):
        self.add_message("assistant", content)

    def set_system_message(self, content: str):
        self.system_message = Message("system", content)

    def reset_messages(self):
        self.messages = []

    def print_messages(self):
        for message in self.messages:
            print(f"{message.role}: {message.content}")

    def debug_log(self, text):
        with io.open(self.debug_log_filename, 'a', encoding='utf-8') as f:
            f.write(text + "\n")