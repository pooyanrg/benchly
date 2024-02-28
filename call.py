import numpy as np
import requests
import pandas as pd
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
import google.generativeai as genai
from google.api_core.exceptions import *
import time

from tqdm import tqdm


retry_strategy = Retry(backoff_factor=2, allowed_methods=frozenset(['GET', 'POST']))
adapter = HTTPAdapter(max_retries=retry_strategy)
session = requests.Session()
session.mount('http://', adapter)
session.mount('https://', adapter)

def get_gpt_payload(model_name, text, image=None):

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


def gemini_call(dataset, model_name, api_key, text_only):

    max_retries = 10 
    base_delay = 1

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)

    all_responses = dict()

    for data in tqdm(dataset):

        question = data["query"]

        for retry_attempt in range(max_retries):
            if not text_only:
                try:
                    response = model.generate_content([question, data["image"]], stream=True)
                    response.resolve()
                    all_responses[data["query_id"]] = response.candidates[0].content.parts[0].text
                    break
                except ServiceUnavailable as e:
                    if retry_attempt < max_retries - 1:
                        delay = base_delay * 2 ** retry_attempt
                        print(f"Retrying in {delay} seconds...")
                        time.sleep(delay)
                except Aborted as e:
                    if retry_attempt < max_retries - 1:
                        delay = base_delay * 2 ** retry_attempt
                        print(f"Retrying in {delay} seconds...")
                        time.sleep(delay)
                except PermissionDenied as e:
                    if retry_attempt < max_retries - 1:
                        delay = base_delay * 2 ** retry_attempt
                        print(f"Retrying in {delay} seconds...")
                        time.sleep(delay)
                except ResourceExhausted as e:
                    if retry_attempt < max_retries - 1:
                        delay = base_delay * 2 ** retry_attempt
                        print(f"Retrying in {delay} seconds...")
                        time.sleep(delay)
            else:
                try:
                    response = model.generate_content(question, stream=True)
                    response.resolve()
                    all_responses[data["query_id"]] = response.candidates[0].content.parts[0].text
                    break
                except ServiceUnavailable as e:
                    if retry_attempt < max_retries - 1:
                        delay = base_delay * 2 ** retry_attempt
                        print(f"Retrying in {delay} seconds...")
                        time.sleep(delay)
                except Aborted as e:
                    if retry_attempt < max_retries - 1:
                        delay = base_delay * 2 ** retry_attempt
                        print(f"Retrying in {delay} seconds...")
                        time.sleep(delay)
                except PermissionDenied as e:
                    if retry_attempt < max_retries - 1:
                        delay = base_delay * 2 ** retry_attempt
                        print(f"Retrying in {delay} seconds...")
                        time.sleep(delay)
                except ResourceExhausted as e:
                    if retry_attempt < max_retries - 1:
                        delay = base_delay * 2 ** retry_attempt
                        print(f"Retrying in {delay} seconds...")
                        time.sleep(delay)

    return all_responses

def gpt_call(dataset, model_name, api_key, text_only):
    
    all_responses = dict()

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    
    for data in tqdm(dataset):
        question = data["query"]
        
        if not text_only:
            payload = get_gpt_payload(model_name, question, data["image"])
        else:
            payload = get_gpt_payload(model_name, question)

        try:
            response = session.post(
                "https://api.openai.com/v1/chat/completions", headers=headers, json=payload
            )
        except:
            return all_responses


        all_responses[data["query_id"]] = response.json()

    return all_responses
      

def mixtral_call(dataset, model_name, api_key, text_only):
    return 0
    


API_LIST = {"gemini": gemini_call,
            "gpt": gpt_call,
            "mixtral": mixtral_call}

