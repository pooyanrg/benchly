import numpy as np
import requests
from bs4 import BeautifulSoup
import pandas as pd
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
import google.generativeai as genai

retry_strategy = Retry(backoff_factor=2)
adapter = HTTPAdapter(max_retries=retry_strategy)
session = requests.Session()
session.mount('http://', adapter)
session.mount('https://', adapter)

def gpt_message(model_name, api_key, text, image=None):

    if image:
        payload = {
            "model": model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image}"},
                        },
                    ],
                }
            ],
            "max_tokens": 2048,
        }
    else:
        payload = {
            "model": model_name,
            "messages": [
                {
                    "role": "user",
                    "content": text,
                },
            ],
            "max_tokens": 2048,
        }
    
    return payload


def gemini_call(inputs, dataset, model_name="gemini-pro", api_key="AIzaSyB2CP7uRo1f0AylHGylS7GkmVApim3-bps"):

    genai.configure(api_key=api_key)

    model = genai.GenerativeModel(model_name)

    all_responses = []

    for data in dataset:
        question = data["question"]

        if "image" in inputs:
            response = model.generate_content([question, data["image"]], stream=True)
        else:
            response = model.generate_content(question, stream=True)
        response.resolve()
        all_responses.append(response.text)

    return all_responses

def gpt_call(inputs, dataset, model_name="gpt-4-0613", api_key="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"):
    
    all_responses = []

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    
    for data in dataset:
         
        question = data["question"]
        
        if "image" == inputs:
            payload = gpt_message(model_name, api_key, question, data["image"])
        else:
            payload = gpt_message(model_name, api_key, question)

        response = requests.post(
            "https://api.openai.com/v1/chat/completions", headers=headers, json=payload
        )
    
        all_responses.append(response.json())

    return all_responses
      

def mixtral_call(inputs, dataset, model_name="mixtral", api_key="joejdwoejowdjd"):
    return 0
    


API_LIST = {"gemini": gemini_call,
            "gpt": gpt_call,
            "mixtral": mixtral_call}

