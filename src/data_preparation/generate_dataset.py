"""Module for cleaning data set/adding embeddings/pushing modified dataset.

Does not need to be run unless you intend on modifying the dataset and 
pushing it to your own Hugging Face account."""
from datasets import load_dataset
import csv
import json

DATA_SET = "mkessle/public-domain-poetry"
EMBEDDING_DATA_SET = "pvd-dot/public-domain-poetry-with-embeddings"
DATA_SET_WITH_EMBEDDINGS_PATH = "data/data_with_embeddings"
EMBEDDINGS_CSV_PATH = "data/embeddings.csv"
EMBEDDINGS_FAISS_PATH = "data/embeddings.faiss"


def main():
    data = load_dataset(DATA_SET, split="train")

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

    data_with_embeddings.push_to_hub(EMBEDDING_DATA_SET)


if __name__ == "__main__":
    main()
