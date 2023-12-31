## Ask the recommender:

Run as streamlit demo:

```
streamlit run src/recommender/streamlit_demo.py
```

Run as CLI tool:

```
python3 src/recommender/main.py
```

## Generate embeddings data and build FAISS index (One time setup):

It costs around $4 to generate embeddings for all 38.5k poems.

```
python3 src/data_preparation/generate_embeddings.py 
python3 src/data_preparation/build_index.py
```


## Installation:

Set your OpenAI API key in a `.env` file (see `.env.example`):

```

Setup virtual environment and install dependencies:

```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

