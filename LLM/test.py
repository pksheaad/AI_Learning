import os
from dotenv import load_dotenv
import openai
from openai import OpenAI

MODEL = "gpt-3.5-turbo"


# loading the api_key
load_dotenv()
api_key = os.getenv(key = "OPENAI_API_KEY")

# create a client
client = OpenAI(api_key = api_key)

messages = [
    {
        "role" : "user",
        "content" : "What is capital of India"
    }
]

response = client.chat.completions.create(
    model = MODEL,
    messages = messages
)

results = response.choices[0].message.content
print(results)