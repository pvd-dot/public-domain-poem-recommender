"""Module contains the LLM prompts and related utilities.

Poem candidates are retrieved using a vector search index containing poem 
embeddings (see vector_searcher.py). The candidate poems are passed into 
the prompt (a technique known as Retrieval-Augmented Generation), along with 
the text of the user request. The LLM is then asked to select the best 
poem candidate, and to provide an explanation for its choice.

The LLM is given examples of how it should answer based on the user query 
and the poem candidates (a technique known as In-Context Learning)."""
import re
import tiktoken

TOKENIZER = "cl100k_base"
POEM_TOKEN_LIMIT = 1000

INITIAL_PROMPT = """
Your job is to recommend one poem from a selected list of poems that would satisfy my preferences.
You must respond with an explanation of why you chose that poem in <explanation> tags, and you must include the id of the selected poem in <id> tags.

Here are some examples:
<example>
user:
<query>
    Recommend me an uplifting poem about winter. 
</query>
<options>
    <poem>
        id: 10
        Title: A Calendar Of Sonnets - January
        Author: Helen Hunt Jackson
        Birth and Death Dates: None

        Views: 892
        Text:     O winter! frozen pulse and heart of fire,
            What loss is theirs who from thy kingdom turn
            Dismayed, and think thy snow a sculptured urn
            Of death! Far sooner in midsummer tire
            The streams than under ice. June could not hire
            Her roses to forego the strength they learn
            In sleeping on thy breast. No fires can burn
            The bridges thou dost lay where men desire
            In vain to build.
                        O Heart, when Love's sun goes
            To northward, and the sounds of singing cease,
            Keep warm by inner fires, and rest in peace.
            Sleep on content, as sleeps the patient rose.
            Walk boldly on the white untrodden snows,
            The winter is the winter's own release.

        About: None
    </poem>
    <poem>
        id: 22
        Title: A Calendar Of Sonnets - February.
        Author: Helen Hunt Jackson
        Birth and Death Dates: None

        Views: 889
        Text:     Still lie the sheltering snows, undimmed and white;
            And reigns the winter's pregnant silence still;
            No sign of spring, save that the catkins fill,
            And willow stems grow daily red and bright.
            These are the days when ancients held a rite
            Of expiation for the old year's ill,
            And prayer to purify the new year's will:
            Fit days, ere yet the spring rains blur the sight,
            Ere yet the bounding blood grows hot with haste,
            And dreaming thoughts grow heavy with a greed
            The ardent summer's joy to have and taste;
            Fit days, to give to last year's losses heed,
            To reckon clear the new life's sterner need;
            Fit days, for Feast of Expiation placed!

        About: None
    </poem>
    <poem>
        id: 98
        Title: A Calendar Of Sonnets - December
        Author: Helen Hunt Jackson
        Birth and Death Dates: None

        Views: 926
        Text:     The lakes of ice gleam bluer than the lakes
            Of water 'neath the summer sunshine gleamed:
            Far fairer than when placidly it streamed,
            The brook its frozen architecture makes,
            And under bridges white its swift way takes.
            Snow comes and goes as messenger who dreamed
            Might linger on the road; or one who deemed
            His message hostile gently for their sakes
            Who listened might reveal it by degrees.
            We gird against the cold of winter wind
            Our loins now with mighty bands of sleep,
            In longest, darkest nights take rest and ease,
            And every shortening day, as shadows creep
            O'er the brief noontide, fresh surprises find.

        About: None
    </poem>
</options>
assistant: 
<explanation>I recommend A Calendar Of Sonnets - February by Helen Hunt Jackson. This poem takes place in the winter months, and there
uplifting ideas in the poem around wiping away the ills of the past year, and looking forward to a fresh start in the new year.</explanation>
<id>22</id>
</example>
"""

RESPONSE_PROMPT = """
<query>
{}
</query>
<options>
{}
</options>
Remember, you must respond with an explanation of why you chose that poem using <explanation> tags, and you must include the id of the selected poem using <id> tags.
Only the explanation will be shown to the user. The id of the poem is internal information and should be removed from the explanation.
"""


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def reduce_to_token_limit(text, limit):
    tokens = num_tokens_from_string(text, TOKENIZER)
    if tokens <= limit:
        return text
    start = 0
    end = len(text)
    while start <= end:
        mid = (start + end) // 2
        if num_tokens_from_string(text[:mid], TOKENIZER) <= limit:
            start = mid + 1
        else:
            end = mid - 1
    return text[:end]


def build_response_prompt(user_query, poem_options):
    poems_text = ""
    for poem in poem_options:
        poem_text = (
            f"id: {poem.id}\n"
            + f"Title: {poem.title}\n"
            + f"Author: {poem.author}\n"
            + f"Birth and Death Dates: {poem.birth_and_death_dates}\n"
            + f"Views: {poem.views}\n"
            + f"Text: {poem.text}\n"
            + f"About: {poem.about}"
        )
        poem_text = reduce_to_token_limit(poem_text, POEM_TOKEN_LIMIT)
        poems_text += f"<poem>\n{poem_text}\n</poem>\n"
    return RESPONSE_PROMPT.format(user_query, poems_text)


def extract_response(response):
    explanation_match = re.search(r"<explanation>(.*)</explanation>", response)
    if explanation_match is None:
        raise ValueError("Response does not contain an explanation.")
    explanation = explanation_match.group(1)

    id_match = re.search(r"<id>(.*)</id>", response)
    if id_match is None:
        raise ValueError("Response does not contain an id.")
    id_ = id_match.group(1)
    return explanation, id_
