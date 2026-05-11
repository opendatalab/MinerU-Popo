import requests
import json
import os
from tqdm import tqdm
from openai import OpenAI

import fitz  # PyMuPDF
from PIL import Image
import io
import base64

# First run Popo and Qwen3-VL Locally by vllm

def popo_generate(prompt, base64_image):

    url = ""
    key = ""
    base_model = "Popo"
    client = OpenAI(
        base_url=url,
        api_key=key
    )
    res = ""
    cnt = 0
    prompt = prompt[:100000] if len(prompt)>100000 else prompt
    
    
    if base64_image:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
    else:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt}
                ],
            }
        ]

    while cnt < 5:
        try:
            response = client.chat.completions.create(
                model=base_model,
                messages=messages,
                max_tokens=50000,
                temperature = 1
            )
            res = response.choices[0].message.content

            return res

        except Exception as e:
            cnt += 1
            print(e)

    return ""

def qwen_generate(prompt, base64_image):

    url = ""
    key = ""
    base_model = "Qwen3-VL-4B-Instruct"
    client = OpenAI(
        base_url=url,
        api_key=key
    )
    res = ""
    cnt = 0
    prompt = prompt[:100000] if len(prompt)>100000 else prompt
    
    
    if base64_image:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
    else:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt}
                ],
            }
        ]

    while cnt < 5:
        try:
            response = client.chat.completions.create(
                model=base_model,
                messages=messages,
                max_tokens=50000,
                temperature = 1
            )
            res = response.choices[0].message.content

            return res

        except Exception as e:
            cnt += 1
            print(e)

    return ""

def gpt_generate(prompt, base64_image):#gemini-3-pro-preview

    url = ""
    key = ""
    base_model = "gemini-3-flash-preview"
    client = OpenAI(
        base_url=url,
        api_key=key
    )
    res = ""
    cnt = 0
    prompt = prompt[:100000] if len(prompt)>100000 else prompt
    
    
    if base64_image:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
    else:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt}
                ],
            }
        ]

    while cnt < 5:
        try:
            response = client.chat.completions.create(
                model=base_model,
                messages=messages,
                temperature = 1
            )
            res = response.choices[0].message.content

            return res

        except Exception as e:
            cnt += 1
            print(e)

    return ""

