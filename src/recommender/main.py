"""Module for running the poem recommender through the CLI.

To run it, run `python3 src/recommender/main.py` from the root directory.
"""
import vector_searcher
import chatgpt
import recommender


def main():
    """Initialize the recommender and ask the user for requests in a loop."""
    vectorsearcher = vector_searcher.VectorSearch()
    chat = chatgpt.ChatGPT()
    recs = recommender.Recommender(vectorsearcher, chat)

    while True:
        query_text = input("User: ")
        explanation, poem_text = recs.ask(query_text)
        print(f"Recommender: {explanation}\n\n{poem_text}")


if __name__ == "__main__":
    main()
