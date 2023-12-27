import vector_searcher
import chatgpt
import recommender

def main():
    vectorsearcher = vector_searcher.VectorSearch()
    chat = chatgpt.ChatGPT()
    recs = recommender.Recommender(vectorsearcher, chat)

    while True:
        query_text = input("User: ")
        results = recs.ask(query_text)
        print(f"Recommender: {results}")

if __name__ == "__main__":
    main()