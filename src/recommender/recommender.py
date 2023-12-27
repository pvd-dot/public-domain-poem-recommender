import prompts

class Recommender:
    def __init__(self, vector_searcher, chat):
        self.vector_searcher = vector_searcher
        self.chat = chat
        self.chat.set_system_message(prompts.INITIAL_PROMPT)

    def build_recommendation_result(self, poem_id, explanation):
        poem = self.vector_searcher.convert_to_poem(poem_id)
        poem_text = f"{poem.title}\n" +\
            f"By {poem.author}\n\n" +\
            f"{poem.text}"
        recommendation_text = f"{explanation}\n\n{poem_text}\n"
        return recommendation_text

    def ask(self, user_query):
        poem_results = self.vector_searcher.search(user_query, limit=10)
        self.chat.add_assistant_message(prompts.build_response_prompt(user_query, poem_results))
        response = self.chat.respond(user_query)
        self.chat.reset_messages()
        try:
            explanation, id = prompts.extract_response(response)
            return self.build_recommendation_result(id, explanation)
        except ValueError as err:
            return "Sorry, please try again with a different query."
