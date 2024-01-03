# Public Domain Poem Recommender

An LLM/KNN-based poem recommender for a collection of 38.5k poems in the public domain. 

<img src="demo.gif" width="550">

## How it works:

 - Embeddings are generated using the poem text and other metadata (author bio, etc) with OpenAI's text embedding model 
 - A FAISS vector search index is built using the embeddings
 - At request time, user recommendation queries are embedded with the same model and used to query the FAISS index for candidate poems 
 - The LLM (ChatGPT) is used to select the most relevant candidate poem 
    - The text of the candidate poems is included in the prompt (Retrieval-Augmented Generation)
    - The LLM is given several examples of how to select poems (In-Context learning)


## Running

Run with GUI using streamlit:

```
streamlit run src/recommender/streamlit_demo.py
```

Run through CLI:

```
python3 src/recommender/main.py
```

Run as Discord bot (requires Discord bot token in `.env` file):

```
python3 src/recommender/discord_bot.py
```

Note: to run the discord bot persistently, you'll need a hosting solution. I deploy bots for my discord server on AWS with ECS, and use S3/EFS for the FAISS index and data set. 

## Set up

Set your OpenAI API key in a `.env` file (see `.env.example`).

Setup virtual environment and install dependencies:

```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Generate embeddings data and build FAISS index (One time setup):

It costs around $4 in OpenAI API costs to generate embeddings for all 38.5k poems.

```
python3 src/data_preparation/generate_embeddings.py 
python3 src/data_preparation/build_index.py
```

## Sample Recommendations Output

```
$ python3 src/recommender/main.py

Welcome to the poem recommender
You can ask the poem recommender anything, and it will try to recommend you a relevant poem.
Enter 'quit' to exit.

User: Can you recommend me a poem about overcoming hardships?
Recommender: I recommend the poem "By Rugged Ways" by Paul Laurence Dunbar. This poem speaks about the struggles and challenges faced on difficult paths, but it also conveys the message of perseverance and hope in the face of adversity.

By Rugged Ways
By Paul Laurence Dunbar

    By rugged ways and thro' the night
    We struggle blindly toward the light;
    And groping, stumbling, ever pray
    For sight of long delaying day.
    The cruel thorns beside the road
    Stretch eager points our steps to goad,
    And from the thickets all about
    Detaining hands reach threatening out.

    "Deliver us, oh, Lord," we cry,
    Our hands uplifted to the sky.
    No answer save the thunder's peal,
    And onward, onward, still we reel.
    "Oh, give us now thy guiding light;"
    Our sole reply, the lightning's blight.
    "Vain, vain," cries one, "in vain we call;"
    But faith serene is over all.

    Beside our way the streams are dried,
    And famine mates us side by side.
    Discouraged and reproachful eyes
    Seek once again the frowning skies.
    Yet shall there come, spite storm and shock,
    A Moses who shall smite the rock,
    Call manna from the Giver's hand,
    And lead us to the promised land!

    The way is dark and cold and steep,
    And shapes of horror murder sleep,
    And hard the unrelenting years;
    But 'twixt our sighs and moans and tears,
    We still can smile, we still can sing,
    Despite the arduous journeying.
    For faith and hope their courage lend,
    And rest and light are at the end.

User: Can you recommend a poem about finding the sublime in everyday life?
Recommender: I recommend "Nature." by John Clare. This poem beautifully expresses the poet's delight in the simple, everyday aspects of nature, showcasing the sublime in the ordinary.

Nature.
By John Clare

    O simple Nature, how I do delight
    To pause upon thy trifles--foolish things,
    As some would call them.--On the summer night,
    Tracing the lane-path where the dog-rose hings
    With dew-drops seeth'd, while chick'ring cricket sings;
    My eye can't help but glance upon its leaves,
    Where love's warm beauty steals her sweetest blush,
    When, soft the while, the Even silent heaves
    Her pausing breath just trembling thro' the bush,
    And then again dies calm, and all is hush.
    O how I feel, just as I pluck the flower
    And stick it to my breast--words can't reveal;
    But there are souls that in this lovely hour
    Know all I mean, and feel whate'er I feel.
    
User: Can you recommend me a poem about weariness and disillusionment?
Recommender: I recommend the poem "Worn Out" by Paul Laurence Dunbar. This poem encapsulates weariness and disillusionment, expressing the feeling of exhaustion from trying and the sense of hopelessness in the face of unfulfilled desires and unending struggles.

Worn Out
By Paul Laurence Dunbar

    You bid me hold my peace
    And dry my fruitless tears,
    Forgetting that I bear
    A pain beyond my years.

    You say that I should smile
    And drive the gloom away;
    I would, but sun and smiles
    Have left my life's dark day.

    All time seems cold and void,
    And naught but tears remain;
    Life's music beats for me
    A melancholy strain.

    I used at first to hope,
    But hope is past and, gone;
    And now without a ray
    My cheerless life drags on.

    Like to an ash-stained hearth
    When all its fires are spent;
    Like to an autumn wood
    By storm winds rudely shent,--

    So sadly goes my heart,
    Unclothed of hope and peace;
    It asks not joy again,
    But only seeks release.