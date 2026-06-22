"""
Code from https://github.com/EnVision-Research/Gaussian-Property
"""

import numpy as np
import os
import random
import base64
import time
from openai import OpenAI, RateLimitError

import requests
import json

random.seed(123)  # Set random seed to 123

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def GPT4V(image_path, prompt, system_prompt=None):
    client = OpenAI(api_key=OPENAI_API_KEY)

    base64_image = encode_image(image_path)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}", "detail": "low"}},
        ],
    })

    for attempt in range(6):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages
            )
            return response.choices[0].message.content.strip()
        except RateLimitError:
            wait = 2 ** attempt
            print(f"Rate limit hit, retrying in {wait}s...")
            time.sleep(wait)

    raise RuntimeError("Exceeded retry attempts due to rate limiting")



# def Qwen(image_path, prompt, system_prompt=None):

#     base64_image = encode_image(image_path)

#     messages = []
#     if system_prompt:
#         messages.append({"role": "system", "content": system_prompt})
#     messages.append({
#         "role": "user",
#         "content": [
#             {"type": "text", "text": prompt},
#             {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
#         ],
#     })

#     response = requests.post(
#         url="https://openrouter.ai/api/v1/chat/completions",
#         headers={
#             "Authorization": f"Bearer {OPENROUTER_API_KEY}",
#             "Content-Type": "application/json",
#         },
#         data=json.dumps({
#             "model": "qwen/qwen-2.5-72b-instruct",
#             "messages": messages,
#         })
#     )
#     return response.json()['choices'][0]['message']['content']

# def Gemini(image_path, prompt, system_prompt=None):

#     base64_image = encode_image(image_path)
#     url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

#     headers = {
#         'Content-Type': 'application/json',
#     }

#     payload = {
#         "contents": [
#             {
#                 "parts": [
#                     {"text": prompt},
#                     {"inline_data": {"mime_type": "image/png", "data": base64_image}},
#                 ]
#             }
#         ]
#     }
#     if system_prompt:
#         payload["system_instruction"] = {"parts": [{"text": system_prompt}]}

#     response = requests.post(url, headers=headers, json=payload)
#     result = response.json()

#     if response.ok:
#         result = response.json()
#         ret = result["candidates"][0]["content"]["parts"][0]["text"]
#         print(ret)
#         return ret
#     else:
#         print("Error:", response.status_code, response.text)
#         raise ValueError

# def GeminiFlash(image_path, prompt, system_prompt=None):
#     base64_image = encode_image(image_path)
#     url = "https://openrouter.ai/api/v1/chat/completions"

#     headers = {
#         "Authorization": f"Bearer {OPENROUTER_API_KEY}",
#         "Content-Type": "application/json"
#     }

#     messages = []
#     if system_prompt:
#         messages.append({"role": "system", "content": system_prompt})
#     messages.append({
#         "role": "user",
#         "content": [
#             {"type": "text", "text": prompt},
#             {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
#         ],
#     })

#     payload = {
#         "model": "google/gemini-2.5-flash-lite",
#         "messages": messages,
#     }

#     response = requests.post(url, headers=headers, json=payload)

#     if response.status_code == 200:
#         result = response.json()
#         return result["choices"][0]["message"]["content"]
#     elif response.status_code == 408:
#         return GeminiFlash(image_path, prompt, system_prompt)
#     else:
#         print("Error:", response.status_code, response.text)
#         raise ValueError

def get_image_files(directory):
    image_files = []
    # Iterate over the directory
    num_image_files = len(os.listdir(directory))
    print("################################################", num_image_files)
    for i in range(1, num_image_files + 1):  # Modify range if more folders need to be processed
        sub_path = str(i).zfill(2)
        now_path = os.path.join(directory, sub_path)
        all_png = os.listdir(now_path)
        for png in all_png:
            img_path = os.path.join(now_path, png)
            image_files.append(img_path)

    image_files = sorted(image_files, key=lambda x: (x.split('/')[-2], x.split('/')[-1]))

    return image_files
