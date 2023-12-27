import collections
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

Poem = collections.namedtuple('Poem', ['id', 'title', 'author', 'text', 'views', 'about', 'birth_and_death_dates'])

class VectorSearch:
    def __init__(self):
        self.client = openai.OpenAI()
        self.data = load_from_disk(DATA_SET_WITH_EMBEDDINGS_PATH)
        self.data.load_faiss_index('embedding', EMBEDDINGS_FAISS_PATH)

    def convert_to_poem(self, id):
        row = self.data[int(id)]
        return Poem(
            id=row['id'],
            title=row['Title'],
            author=row['Author'],
            text=row['Poem Text'],
            views=row['Views'],
            about=row['About'],
            birth_and_death_dates=row['Birth and Death Dates']
        )

    def search(self, query_text, limit=1):
        query_embedding = np.array(self.client.embeddings.create(input=[query_text], model=MODEL).data[0].embedding)
        _, results = self.data.search('embedding', query_embedding, k=limit)
        return [self.convert_to_poem(id) for id in results]