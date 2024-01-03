"""Module for cleaning data set and building a FAISS index of poem embeddings."""
from datasets import load_dataset
import csv
import json

DATA_SET = "mkessle/public-domain-poetry"
DATA_SET_WITH_EMBEDDINGS_PATH = "data/data_with_embeddings"
EMBEDDINGS_CSV_PATH = "data/embeddings.csv"
EMBEDDINGS_FAISS_PATH = "data/embeddings.faiss"

MAX_EMBEDDINGS = 38521


def main():
    data = load_dataset(DATA_SET, split=f"train[:{MAX_EMBEDDINGS}]")

    ids = list(range(len(data)))
    data = data.add_column("id", ids)

    embeddings_dct = {}
    with open(EMBEDDINGS_CSV_PATH, "r", encoding="UTF-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            embeddings_dct[int(row["id"])] = json.loads(row["embedding"])

    data_with_embeddings = data.map(
        lambda row: {
            "embedding": embeddings_dct[row["id"]],
            "Poem Text": str(row["Poem Text"]).replace(
                "�", "'"  # original dataset has some faulty encodings of "'"
            ),
        },
    )

    data_with_embeddings.save_to_disk(DATA_SET_WITH_EMBEDDINGS_PATH)

    data_with_embeddings.add_faiss_index(column="embedding")
    data_with_embeddings.save_faiss_index("embedding", "data/embeddings.faiss")


if __name__ == "__main__":
    main()
