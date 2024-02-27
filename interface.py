import argparse
import json
import os
from datasets import load_dataset

from call import API_LIST

def get_args(description='Bencly on LLM/VLMs'):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--judge', action='store_true', help="Whether to use for judge.")
    parser.add_argument('--llm', action='store_true', help="Whether to use text inputs only.")
    parser.add_argument("--model", type=str, default='gpt-4-0613', help='which model to use')
    parser.add_argument('--family', type=str, default='gpt', help='family for api key retrieval')

    parser.add_argument('--config', type=str, default='config.json', help='config file path')
    parser.add_argument('--output_dir', type=str, default='ckpts/', help='output directory')
    parser.add_argument('--data_path', type=str, default='data/datapath', help='dataset directory')

    args = parser.parse_args()

    return args

def save_results(output_dir, model_name, responses):

    path = os.path.join(output_dir, model_name + "_responses.json")
    if os.path.isfile(path):
        print("results already exist!")
    else:
        with open(path, 'wb') as fp:
            json.dump(responses)

def main():
    args = get_args()

    with open(args.config, 'r') as fp:
        config = json.load(fp)
    
    api_key = config["keys"][args.family]
    api = API_LIST[args.family]
    
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    
    if os.path.isfile(args.data_path):
        with open(args.data_path, 'r') as fp:
            dataset = json.load(fp)
    else:
        dataset = load_dataset(config["dataset"])

    all_responses = api(dataset, args.model, api_key, args.llm)

    save_results(args.output_dir, args.model, all_responses)


if __name__ == "__main__":
    main()

