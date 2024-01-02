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
<example>
user:
<query>
    Recommend me a poem about longing for the past.
</query>
<options>
<poem>
id: 26248
Title: Reminiscence
Author: John Charles McNeill
Birth and Death Dates: 1874 - 1907
Views: 1028
Text:              We sang old love-songs on the way
                In sad and merry snatches,
             Your fingers o'er the strings astray
                Strumming the random catches.

             And ever, as the skiff plied on
                Among the trailing willows,
             Trekking the darker deeps to shun
                The gleaming sandy shallows,

             It seemed that we had, ages gone,
                In some far summer weather,
             When this same faery moonlight shone,
                Sung these same songs together.

             And every grassy cape we passed,
                And every reedy island,
             Even the bank'd cloud in the west
                That loomed a sombre highland;

             And you, with dewmist on your hair,
                Crowned with a wreath of lilies,
             Laughing like Lalage the fair
                And tender-eyed like Phyllis:

             I know not if 't were here at home,
                By some old wizard's orders,
             Or long ago in Crete or Rome
                Or fair Provencal borders,

             But now, as when a faint flame breaks
                From out its smouldering embers,
             My heart stirs in its sleep, and wakes,
                And yet but half-remembers

             That you and I some other time
                Moved through this dream of glory,
             Like lovers in an ancient rhyme,
                A long-forgotten story.
About:  About: 
</poem>
<poem>
id: 19227
Title: Memories Of Schooldays.
Author: Thomas Frederick Young
Birth and Death Dates: ?  - 1940
Views: 916
Text:     There are mem'ries glad of the old school-house,
     Which throng around me still;
    And voices spoke in my youthful days,
     My ears with music fill.

    Those youthful voices I seem to hear,
     With their gladsome, joyous tone,
    And joy and hope they bring to me,
     When I am all alone.

    I think of the joys of that time long past,
     Of its boyish hopes and fears,
    And 'tis partly joy, and partly pain,
     That wets my eyes with tears.

    For 'tis joy I feel, when I seem to stand,
     Where I stood long years ago,
    And when I think that cannot be,
     My heart is fill'd with woe.

    My old school mates are scatter'd far,
     And some are with the dead,
    And my old class mates have wander'd, too,
     To seek for fame, or bread.

    And those who still are near my home,
     And whom I often see,
    Have come to manhood's grave estate;
     They're boys no more to me.

    And tho' we meet in converse yet,
     And each one's thoughts enjoy,
    Our thoughts and words are not so free,
     As when, each was a boy.

    For the spring of life is gone for us,
     With all its bursting bloom,
    And manhood's thoughts, and joys, and cares,
     Are now within its room.

    But the mem'ry of our bright school days,
     Will last through ev'ry strain,
    And time will brighten ev'ry joy,
     And darken ev'ry pain.

    The rippling of our childhood's laugh,
     Will roll adown the years,
    And time will blunt, each day we live,
     The mem'ry of our tears.

    Our boyhood's hopes, and boyhood's dreams,
     And aspirations high,
    Will doubtless never be fulfill'd,
     Until the day we die.

    But still we'll cherish in our hearts,
     And live those days again,
    When awkardly we read our books,
     Or trembling held the pen.
About: Canadian Poet
</poem>
<poem>
id: 17731
Title: The Long Ago.
Author: Jean Blewett
Birth and Death Dates: 1862 - 1934
Views: 1095
Text:         O life has its seasons joyous and drear,
         Its summer sun and its winter snow,
        But the fairest of all, I tell you, dear,
         Was the sweet old spring of the long ago - 
                 The ever and ever so long ago - 

        When we walked together among the flowers,
         When the world with beauty was all aglow.
        O the rain and dew! O the shine and showers
         Of the sweet old spring of the long ago!
                 The ever and ever so long ago.

        A hunger for all of the past delight
         Is stirred by the winds that softly blow.
        Can you spare me a thought from heaven to-night
         For the sweet old spring of the long ago? - 
                 The ever and ever so long ago.
About: 
</poem>
<poem>
id: 13409
Title: Home Yearnings
Author: John Clare
Birth and Death Dates: 13 July 1793 � 20 May 1864
Views: 1711
Text:      O for that sweet, untroubled rest
     That poets oft have sung!--
     The babe upon its mother's breast,
     The bird upon its young,
     The heart asleep without a pain--
     When shall I know that sleep again?

     When shall I be as I have been
     Upon my mother's breast--
     Sweet Nature's garb of verdant green
     To woo to perfect rest--
     Love in the meadow, field, and glen,
     And in my native wilds again?

     The sheep within the fallow field,
     The herd upon the green,
     The larks that in the thistle shield,
     And pipe from morn to e'en--
     O for the pasture, fields, and fen!
     When shall I see such rest again?

     I love the weeds along the fen,
     More sweet than garden flowers,
     For freedom haunts the humble glen
     That blest my happiest hours.
     Here prison injures health and me:
     I love sweet freedom and the free.

     The crows upon the swelling hills,
     The cows upon the lea,
     Sheep feeding by the pasture rills,
     Are ever dear to me,
     Because sweet freedom is their mate,
     While I am lone and desolate.

     I loved the winds when I was young,
     When life was dear to me;
     I loved the song which Nature sung,
     Endearing liberty;
     I loved the wood, the vale, the stream,
     For there my boyhood used to dream.

     There even toil itself was play;
     'T was pleasure e'en to weep;
     'T was joy to think of dreams by day,
     The beautiful of sleep.
     When shall I see the wood and plain,
     And dream those happy dreams again?
About:  About: John Clare was an English poet, in his time he was commonly known as "the Northamptonshire Peasant Poet".
</poem>
</options>
<explanation>I recommend Home Yearnings by John Clare. This poem captures the author's nostalgia for free and peaceful days 
spent growing up in the countryside.</explanation>
<id>13409</id>
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
