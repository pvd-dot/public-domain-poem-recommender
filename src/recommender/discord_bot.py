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

description = (
    "A bot for recommending poems in the public domain. Type !recpoem"
    + "followed by a request to get a poem recommendation."
)

intents = discord.Intents.default()
intents.message_content = (
    True  # This bot requires 'message_content' privileged intents to be enabled.
)

bot = commands.Bot(command_prefix="!", description=description, intents=intents)
vectorsearcher = vector_searcher.VectorSearch()
chat = chatgpt.ChatGPT()
recs = recommender.Recommender(vectorsearcher, chat)


@bot.command()
async def recpoem(ctx, *, user_request: str):
    explanation, poem_text = recs.ask(user_request)
    await ctx.send(f"{explanation}\n```{poem_text}```")


bot.run(bot_token)
