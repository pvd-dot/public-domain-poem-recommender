import os
import dotenv
from datasets import load_dataset
import openai
import csv
import json

DATA_SET = "mkessle/public-domain-poetry"
DATA_SET_WITH_EMBEDDINGS_PATH = "data/data_with_embeddings"
EMBEDDINGS_CSV_PATH = "data/embeddings.csv"
EMBEDDINGS_FAISS_PATH = "data/embeddings.faiss"

NUM_THREADS = 5
MAX_EMBEDDINGS = 100
MODEL = "text-embedding-ada-002"

dotenv.load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

client = openai.OpenAI()
data = load_dataset(DATA_SET, split=f"train[:{MAX_EMBEDDINGS}]")

ids = list(range(len(data)))
data = data.add_column('id', ids)

embeddings_dct = {}
with open(EMBEDDINGS_CSV_PATH, 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        embeddings_dct[int(row['id'])] = json.loads(row['embedding'])

data_with_embeddings = data.map(lambda row: {'embedding': embeddings_dct[row['id']]})

data_with_embeddings.save_to_disk('data/data_with_embeddings')

data_with_embeddings.add_faiss_index(column='embedding')
data_with_embeddings.save_faiss_index('embedding', 'data/embeddings.faiss')

