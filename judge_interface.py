import argparse
import json
import os

from call import API_LIST
from call import JUDGE_API_LIST

def get_args(description='Bencly Judge Evaluation'):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--model", type=str, default='gpt-4-0613', help='which model to use')
    parser.add_argument('--family', type=str, default='gpt', help='family for api key retrieval')

    parser.add_argument('--config', type=str, default='config.json', help='config file path')
    parser.add_argument('--input_file', type=str, default='ckpts/gemini-pro_responses.json', help='input directory')
    parser.add_argument('--output_dir', type=str, default='results/', help='output directory')

    args = parser.parse_args()

    return args


def save_results(output_dir, model_name, responses):

    path = os.path.join(output_dir, model_name + "_judged.json")

    if os.path.isfile(path):
        print("results already exist!")
    else:
        with open(path, 'w') as fp:
            json.dump(responses, fp)
        print("results  saved!")

def main():
    args = get_args()

    with open(args.config, 'r') as fp:
        config = json.load(fp)
    
    api_key = config["keys"][args.family]
    api = JUDGE_API_LIST[args.family]

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    if os.path.isfile(args.input_file):
        with open(args.input_file, 'r') as fp:
            responses = json.load(fp)

    question = config["template_judge"]

    all_responses = api(question, responses, args.model, api_key)

    save_results(args.output_dir, args.model, all_responses)


if __name__ == "__main__":
    main()