import prompts

class Recommender:
    def __init__(self, vector_searcher, chat):
        self.vector_searcher = vector_searcher
        self.chat = chat
        self.chat.add_system_message(prompts.INITIAL_PROMPT)

    def ask(self, user_text):
        search_results = self.vector_searcher.search(user_text, limit=10)
        related_texts = []
        for result in search_results:
            related_texts.append(f"Title: {result['Title']}")
        related_texts = "\n\n".join(related_texts)
        response_prompt=prompts.RESPONSE_PROMPT.format(related_texts)
        self.chat.add_assistant_message(response_prompt)
        response = self.chat.respond(user_text)
        return response    
