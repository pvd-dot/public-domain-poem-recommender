import vector_searcher
import chatgpt
import recommender

def main():
    vectorsearcher = vector_searcher.VectorSearch()
    chat = chatgpt.ChatGPT()
    recs = recommender.Recommender(vectorsearcher, chat)

    while True:
        query_text = input("User: ")
        explanation, poem_text = recs.ask(query_text)
        print(f"Recommender: {explanation}\n\n{poem_text}")

if __name__ == "__main__":
    main()