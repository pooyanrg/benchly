from litellm import completion
import argparse
import os
import json
import logging

from datasets import load_dataset
from tqdm import tqdm
import numpy as np

def get_args(description='Benchly Judge Evaluation'):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--llm', action='store_true', help="Whether to use text inputs only.")
    parser.add_argument('--judge', type=int, default=0, help="Whether to judge the response.")
    parser.add_argument("--model", type=str, default='gpt-4-0613', help='which model to use')
    parser.add_argument("--family", type=str, default='gemini', help='which model to use')
    parser.add_argument('--seed', type=int, default=42, help='Whether to evaluate random samples')
    parser.add_argument('--seed_size', type=int, default=5, help='Number of random samples')
    parser.add_argument('--num_retries', type=int, default=10, help='Number of retires for prompting')
    parser.add_argument('--diff_levels', type=list, default=[1,2,3,4,5], help='difficulty levels to evaluate')

    parser.add_argument('--config', type=str, default='config.json', help='config file path')
    parser.add_argument('--experiment', type=str, default='ckpts/', help='experiment directory')

    args = parser.parse_args()

    return args

def get_logger(filename=None):
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format='%(asctime)s - %(levelname)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
    if filename is not None:
        handler = logging.FileHandler(filename)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logging.getLogger().addHandler(handler)
    return logger

def save_results(path, response):

    def convert_numpy_bool(obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        raise TypeError

    with open(path, 'w') as fp:
        json.dump(response, fp, default=convert_numpy_bool)

def result_exists(path):
    if os.path.exists(path):
        with open(path, 'r') as fp:
            temp = json.load(fp)
        if len(temp.keys()) > 0:
            return 1
    return 0

def make_all(path, temp_path):

    responses = dict()

    for file in sorted(os.listdir(temp_path)):
        with open(os.path.join(temp_path, file), 'r') as fp:
            response = json.load(fp)
        responses[response['query_id']] = response

    if not result_exists(path):
        save_results(path, responses)

def get_message(prompt, image_url=None):
    if image_url:
        message = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt
                },
                {
                    "type": "image_url",
                    "image_url": {"url": image_url}
                    }
                ]
            }
        ]
    
    else:
        message = [
            {
            "role": "user",
            "content": prompt
            }
        ]
    
    return message

def api_handler(model, dataset, text_only, path, num_retries):

    for i in tqdm(range(len(dataset))):

        response_dict = dict()
        response_dict['query'] = dataset.iloc[i]["query"]
        response_dict['query_id'] = dataset.iloc[i]["query_id"]
        response_dict['gt_answer'] = dataset.iloc[i]["answer"]

        if text_only:
            query = get_message(dataset.iloc[i]["query"])
        else:
            query = get_message(dataset.iloc[i]["query"], dataset.iloc[i]["image"])

        response = completion(model=model, messages=query, num_retries=num_retries)

        response_dict['response'] = response

        save_path = os.path.join(path, str(dataset.iloc[i]["query_id"])) + '.json'

        save_results(save_path, response_dict)

def api_handler_judge(model, dataset, path, num_retries, question):

    for id, value in tqdm(dataset.items()):

        response_dict = dict()
        response_dict['query'] = question
        response_dict['query_id'] = id

        response_content = (
                value["response"]["choices"][0]["message"]["content"]
                if "response" in value and "choices" in value["response"]
                else None
            )

        temp_values = dict({'model_output': response_content, 'gt_answer':value['gt_answer']})
        query = question.format(**temp_values)

        query = get_message(query)

        response = completion(model=model, messages=query, num_retries=num_retries)

        response_dict['judge_response'] = response

        save_path = os.path.join(path, str(id) + '_judged.json')

        save_results(save_path, response_dict)

def main():

    args = get_args()

    model = args.model
    
    with open(args.config, 'r') as fp:
        config = json.load(fp)

    temp_path = args.experiment + '/temp'

    if not os.path.isdir(temp_path):
        os.makedirs(temp_path)

    logger = get_logger(os.path.join(args.experiment, "log.txt"))
    logger.info("Prompting LLM/VLMs experiment details:")
    logger.info('\t>>>model: {}'.format(model))
    logger.info('\t>>>experiment directory: {}'.format(args.experiment))
    
    dataset = load_dataset(config["dataset"])["validation"].to_pandas()

    diff_levels_int = [int(level) for level in args.diff_levels if level != ',']
    dataset = dataset[dataset['difficulty_level'].isin(diff_levels_int)]
    
    if args.seed > 0:
        dataset = dataset.sample(n=args.seed_size, random_state=args.seed)
    logger.info('\t>>>dataset: {}'.format(config["dataset"]))
    logger.info('\t>>>seed: {}'.format(args.seed))
    logger.info('\t>>>random size: {}'.format(args.seed_size))
    logger.info('\t>>>text mode: {}'.format(args.llm))
    logger.info('\t>>>difficulty level: {}'.format(diff_levels_int))

    api_handler(model, dataset, args.llm, temp_path, args.num_retries)
    
    path = os.path.join(args.experiment, args.family + '.json')

    make_all(path, temp_path)

    if args.judge:
        logger.info("\n\n\n")
        logger.info("Judging LLM/VLMs experiment details:")

        with open(path, 'r') as fp:
            dataset = json.load(fp)
        
        question = config["template_judge"]
        logger.info('\t>>>query: {}'.format(question))

        api_handler_judge(model, dataset, temp_path, args.num_retries, question)

        path = os.path.join(args.experiment, args.family + '_judged.json')
        make_all(path, temp_path)


if __name__ == "__main__":
    main()

