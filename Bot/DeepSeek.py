from huggingface_hub import InferenceClient
import os
from dotenv import load_dotenv, find_dotenv
import json

load_dotenv(find_dotenv())
hg_token = os.environ.get("HUGGING_FACE_TOKEN")

client = InferenceClient(
    # provider="https://api.nebius.ai/huggingface",
    # api_key=hg_token
    provider='together',
    model='deepseek-ai/DeepSeek-V3',
    token=hg_token
)

def deepseek_answer(system_prompt, user_message, client=client):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]

    completion = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-V3",
        messages=messages,
        max_tokens=128,
    )

    return completion.choices[0].message.content

def answ2dict(text):
    try:
        data = json.loads(text)
    except:
        print('bad json convert')
        data = 'repeat prompt'

    return data