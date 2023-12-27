## Ask the recommender:

```
python3 src/recommender/main.py
```

## Generate embeddings data and build FAISS index (One time setup):

```
python3 src/data_preparation/generate_embeddings.py
python3 src/data_preparation/build_index.py
```


## Installation:

```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
