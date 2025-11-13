import os
from openai import OpenAI
os.environ["ARK_API_KEY"] = ""
client = OpenAI(
    base_url="",
    api_key=os.environ.get("ARK_API_KEY"),
)

def openai_chat(prompt):
    completion = client.chat.completions.create(
    model='',
    messages=[
        {"role": "system", "content": "You are a helpful, respectful and honest assistant."},
        {"role": "user", "content": prompt},
    ],
)

    return completion.choices[0].message.content
