"""Module for running the recommender as a Discord bot."""
import discord
import os
from discord.ext import commands
import dotenv

import vector_searcher
import chatgpt
import recommender

dotenv.load_dotenv()
bot_token = os.getenv("DISCORD_BOT_TOKEN")
DISCORD_MESSAGE_LIMIT = 2000
BACKTICKS_BUFFER = 6

description = (
    "A bot for recommending poems in the public domain. Type !recpoem"
    + "followed by a request to get a poem recommendation."
)

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix="!", description=description, intents=intents)
vectorsearcher = vector_searcher.VectorSearch()
chat = chatgpt.ChatGPT()
recs = recommender.Recommender(vectorsearcher, chat)


@bot.command()
async def recpoem(ctx, *, user_request: str):
    explanation, poem_text = recs.ask(user_request)

    def chunk_poem_text(text, lim):
        if lim <= BACKTICKS_BUFFER:
            return "", text
        lim -= BACKTICKS_BUFFER
        if "\n" not in text and len(text) > lim:
            return f"```{text[:lim]}```", text[lim:]
        chunk = ""
        next_ = 0
        while text and len(chunk + text[:next_]) <= lim:
            chunk += text[:next_]
            text = text[next_:]
            try:
                next_ = text.index("\n") + 1
            except ValueError:
                next_ = len(text)
        if chunk:
            return f"```{chunk}```", text
        else:
            return "", text

    while explanation and len(explanation) > DISCORD_MESSAGE_LIMIT:
        await ctx.send(explanation[:DISCORD_MESSAGE_LIMIT])
        explanation = explanation[DISCORD_MESSAGE_LIMIT:]

    chunk, poem_text = chunk_poem_text(
        poem_text, DISCORD_MESSAGE_LIMIT - len(explanation)
    )
    await ctx.send(f"{explanation}{chunk}")

    while poem_text:
        chunk, poem_text = chunk_poem_text(poem_text, DISCORD_MESSAGE_LIMIT)
        await ctx.send(chunk)


bot.run(bot_token)
