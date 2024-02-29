import numpy as np
import requests
import pandas as pd
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
import google.generativeai as genai
from google.api_core.exceptions import *
import time
import os
import json

from tqdm import tqdm

retry_strategy = Retry(backoff_factor=2, allowed_methods=frozenset(['GET', 'POST']))
adapter = HTTPAdapter(max_retries=retry_strategy)
session = requests.Session()
session.mount('http://', adapter)
session.mount('https://', adapter)


def save_results(output_dir, model_name, response):

    path = os.path.join(output_dir, model_name + "response.json")

    with open(path, 'w') as fp:
        json.dump(response, fp)

    print("results  saved!")

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


def gemini_call(dataset, model_name, api_key, path, text_only=True):

    max_retries = 10 
    base_delay = 1

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)

    for i in tqdm(range(len(dataset))):

        response_dict = dict()
        response_dict['query_id'] = dataset.iloc[i]["query_id"]

        question = dataset.iloc[i]["query"]
        response_dict['query'] = question
        response_dict['gt_answer'] = dataset.iloc[i]["answer"]

        for retry_attempt in range(max_retries):
            if not text_only:
                try:
                    response = model.generate_content([question, dataset.iloc[i]["image"]], stream=True)
                    response.resolve()
                    response_dict['response'] = response.candidates[0].content.parts[0].text
                    save_results(path,  model_name + '_' + str(response_dict["query_id"]) + '_', response_dict)
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
                    response_dict['response'] = response.candidates[0].content.parts[0].text
                    save_results(path,  model_name + '_' + str(response_dict["query_id"]) + '_', response_dict)
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
        
        if retry_attempt > max_retries:
            print("server error!")

def gpt_call(dataset, model_name, api_key, path, text_only=True):

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    
    for i in tqdm(range(len(dataset))):

        response_dict = dict()
        response_dict['query_id'] = dataset.iloc[i]["query_id"]

        question = dataset.iloc[i]["query"]
        response_dict['query'] = question
        response_dict['gt_answer'] = dataset.iloc[i]["answer"]
        
        if not text_only:
            payload = get_gpt_payload(model_name, question, dataset.iloc[i]["image"])
        else:
            payload = get_gpt_payload(model_name, question)

        try:
            response = session.post(
                "https://api.openai.com/v1/chat/completions", headers=headers, json=payload
            )
            response_dict['response'] = response.json()
            save_results(path,  model_name + '_' + str(response_dict["query_id"]) + '_', response_dict)
        except:
            print("request error!")
      

def mixtral_call(dataset, model_name, api_key, path, text_only=True):
    
    if not text_only:
        print("Mixtral is not a VLM!")
        return {}


    headers = {"Content-Type": "application/json"}
    
    for i in tqdm(range(len(dataset))):

        response_dict = dict()
        response_dict["query_id"] = dataset.iloc[i]["query_id"]

        question = dataset.iloc[i]["query"]
        response_dict['query'] = question
        response_dict['gt_answer'] = dataset.iloc[i]["answer"]

        data = {"model": "mixtral", "messages": [{"role": "user", "content": question}]}
        
        try:
            response = session.post(
                "http://localhost:11434/api/chat", headers=headers, json=data
            )
            response_dict['response'] = response.json()
            save_results(path,  model_name + '_' + str(response_dict["query_id"]) + '_', response_dict)
        except:
            print("request error!")
    
def gemini_judge(question, responses, model_name, api_key, path):

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)


    for id, value in tqdm(responses.items()):
        response_dict = dict()
        response_dict['query'] = question
        response_dict['query_id'] = id
        
        temp_values = dict({'model_output': value['response'], 'gt_answer':value['gt_answer']})
        query = question.format(**temp_values)

        response = model.generate_content(query, stream=True)
        response.resolve()
        response_dict['judge_response'] = response.candidates[0].content.parts[0].text

        save_results(path, model_name + '_' + str(id) + '_', response_dict)

def gpt_judge(question, responses, model_name, api_key, path):

    response_dict = dict()

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    for id, value in tqdm(responses.items()):

        response_dict = dict()
        response_dict['query'] = question
        response_dict['query_id'] = id
        
        temp_values = dict({'model_output': value['response'], 'gt_answer':value['gt_answer']})
        query = question.format(**temp_values)
        
        payload = get_gpt_payload(model_name, query)

        try:
            response = session.post(
                "https://api.openai.com/v1/chat/completions", headers=headers, json=payload
            )

            response_dict['judge_response'] = response.json()

            save_results(path, model_name + '_' + str(id) + '_', response_dict)
        except:
            print("server error!")

def mixtral_judge(question, responses, model_name, api_key, path):

    headers = {"Content-Type": "application/json"}

    for id, value in tqdm(responses.items()):
        response_dict = dict()
        response_dict['query'] = question
        response_dict['query_id'] = id
        
        temp_values = dict({'model_output': value['response'], 'gt_answer':value['gt_answer']})
        query = question.format(**temp_values)
        
        data = {"model": "mixtral", "messages": [{"role": "user", "content": query}]}

        try:
            response = session.post(
                "http://localhost:11434/api/chat", headers=headers, json=data
            )
            response_dict['judge_response'] = response.json()

            save_results(path, model_name + '_' + str(id) + '_', response_dict)
        except:
            print("server error!")



API_LIST = {"gemini": gemini_call,
            "gpt": gpt_call,
            "mixtral": mixtral_call}

JUDGE_API_LIST = {"gemini": gemini_judge,
                  "gpt": gpt_judge,
                  "mixtral": mixtral_judge}

