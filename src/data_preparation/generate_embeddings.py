import os
import threading
import concurrent.futures
import dotenv
from datasets import load_dataset
import openai
import csv
import json

DATA_SET = "mkessle/public-domain-poetry"
EMBEDDINGS_CSV_PATH = "data/embeddings.csv"

NUM_THREADS = 5
MAX_EMBEDDINGS = 100
MODEL = "text-embedding-ada-002"

dotenv.load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

client = openai.OpenAI()
data = load_dataset(DATA_SET, split=f"train[:{MAX_EMBEDDINGS}]")
lock = threading.Lock()
seen = set([])

def build_text(row):
    text = ""
    for col in row:
        val = row[col]
        # data set specific handling - clean up breaking spaces and redundant headers
        if isinstance(val, str):
            val = val.replace(u'\xa0', u' ').strip()
            if val[:len(col)] == col:
                val = val[len(col):].strip()
            if val[:1] == ":":
                val = val[1:].strip()
            val = val.strip()
        text += f"{col}: {val}\n"
    return text

def generate_embedding(worker_id, client, writer, seen):
    for k in range(len(data)):
        if k % NUM_THREADS == worker_id and k not in seen:
            try:
                embedding = client.embeddings.create(input=[build_text(data[k])], model=MODEL).data[0].embedding
                with lock:
                    writer.writerow({'id': k, 'embedding': json.dumps(embedding)})
                    seen.add(k)

            except Exception as exc:
                print(f'worker {worker_id} generated an exception for row {k}: {exc}')

with open(EMBEDDINGS_CSV_PATH, 'a+') as file:
    newfile = os.path.getsize(EMBEDDINGS_CSV_PATH) == 0
    if not newfile:
        file.seek(0)
        reader = csv.DictReader(file)
        for row in reader:
            seen.add(int(row['id']))

    writer = csv.DictWriter(file, fieldnames=['id', 'embedding'])
    if newfile:
        writer.writeheader()

    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        future_to_worker = {executor.submit(generate_embedding, worker_id, client, writer, seen): worker_id for worker_id in range(NUM_THREADS)}

        for future in concurrent.futures.as_completed(future_to_worker):
            worker_id = future_to_worker[future]
            try:
                future.result()
            except Exception as exc:
                print(f'worker {worker_id} generated an exception: {exc}')


