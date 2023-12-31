"""Module for generating embeddings for the poems in the data set.

This module appends to a CSV file of poem embeddings. It will not 
regenerate embeddings already present in the CSV. It runs workers on
multiple threads to speed up the embedding generation process. If it is 
interrupted or any errors are encountered, it can be re-run to generate 
the remaining embeddings without re-doing work.
"""
import os
import tiktoken
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
MAX_EMBEDDINGS = 38521
MODEL = "text-embedding-ada-002"
TOKENIZER = "cl100k_base"
TOKEN_LIMIT = 8192

dotenv.load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

client = openai.OpenAI()
data = load_dataset(DATA_SET, split=f"train[:{MAX_EMBEDDINGS}]")
lock = threading.Lock()
seen = set([])


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def reduce_to_token_limit(text):
    tokens = num_tokens_from_string(text, TOKENIZER)
    if tokens <= TOKEN_LIMIT:
        return text
    start = 0
    end = len(text)
    while start <= end:
        mid = (start + end) // 2
        if num_tokens_from_string(text[:mid], TOKENIZER) <= TOKEN_LIMIT:
            start = mid + 1
        else:
            end = mid - 1
    return text[:end]


def build_text(row):
    text = ""
    for col in row:
        val = row[col]
        # Data set specific handling for cleaning up spaces and
        # redundant headers.
        if isinstance(val, str):
            val = val.replace("\xa0", " ").strip()
            if val[: len(col)] == col:
                val = val[len(col) :].strip()
            if val[:1] == ":":
                val = val[1:].strip()
            val = val.strip()
        text += f"{col}: {val}\n"

    return reduce_to_token_limit(text)


def generate_embedding(worker_id, writer):
    count = 0
    failures = 0
    total = sum(
        [1 for k in range(len(data)) if k % NUM_THREADS == worker_id and k not in seen]
    )
    print(f"worker {worker_id} has {total} remaining embeddings to generate.")
    for k in range(len(data)):
        if k % NUM_THREADS == worker_id and k not in seen:
            try:
                embedding = (
                    client.embeddings.create(input=[build_text(data[k])], model=MODEL)
                    .data[0]
                    .embedding
                )
                with lock:
                    writer.writerow({"id": k, "embedding": json.dumps(embedding)})
                    seen.add(k)
                    count += 1
            except Exception as exc:  # pylint: disable=broad-except
                print(
                    f"worker {worker_id} generated an exception"
                    + f" for row {k} (poem: data[]): {exc}"
                )
                failures += 1
            if count > 0 and count % 1000 == 0:
                print(
                    f"worker {worker_id} has generated {count} embeddings"
                    + f" out of {total} with {failures} failures."
                )

    print(
        f"worker {worker_id} has finished generating all {count} remaining embeddings."
    )


def generate_all_embeddings():
    with open(EMBEDDINGS_CSV_PATH, "a+", encoding="UTF-8") as file:
        newfile = os.path.getsize(EMBEDDINGS_CSV_PATH) == 0
        if not newfile:
            file.seek(0)
            reader = csv.DictReader(file)
            for row in reader:
                seen.add(int(row["id"]))

        writer = csv.DictWriter(file, fieldnames=["id", "embedding"])
        if newfile:
            writer.writeheader()

        with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
            future_to_worker = {
                executor.submit(generate_embedding, worker_id, writer): worker_id
                for worker_id in range(NUM_THREADS)
            }

            for future in concurrent.futures.as_completed(future_to_worker):
                worker_id = future_to_worker[future]
                try:
                    future.result()
                except Exception as exc:  # pylint: disable=broad-except
                    print(f"worker {worker_id} generated an exception: {exc}")


def main():
    generate_all_embeddings()


if __name__ == "__main__":
    main()
