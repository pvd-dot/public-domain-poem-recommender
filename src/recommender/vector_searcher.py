import openai
import numpy as np
from datasets import load_from_disk
import dotenv
import os

DATA_SET_WITH_EMBEDDINGS_PATH = "data/data_with_embeddings"
EMBEDDINGS_FAISS_PATH = "data/embeddings.faiss"

MODEL = "text-embedding-ada-002"

dotenv.load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

class VectorSearch:
    def __init__(self):
        self.client = openai.OpenAI()
        self.data = load_from_disk(DATA_SET_WITH_EMBEDDINGS_PATH)
        self.data.load_faiss_index('embedding', EMBEDDINGS_FAISS_PATH)
       
    def search(self, query_text, limit=1):
        query_embedding = np.array(self.client.embeddings.create(input=[query_text], model=MODEL).data[0].embedding)
        _, results = self.data.search('embedding', query_embedding, k=limit)
        return [self.data[int(k)] for k in results]