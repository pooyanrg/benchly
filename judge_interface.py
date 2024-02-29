import argparse
import json
import os

from call import JUDGE_API_LIST

def get_args(description='Bencly Judge Evaluation'):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--model", type=str, default='gpt-4-0613', help='which model to use')
    parser.add_argument('--family', type=str, default='gpt', help='family for api key retrieval')

    parser.add_argument('--config', type=str, default='config.json', help='config file path')
    parser.add_argument('--input_dir', type=str, default='ckpts/', help='input directory')
    parser.add_argument('--output_dir', type=str, default='results/', help='output directory')

    args = parser.parse_args()

    return args

def main():
    args = get_args()

    with open(args.config, 'r') as fp:
        config = json.load(fp)

    if os.path.isdir(args.output_dir):
        print("results file already exists! Change the output directory.")
        
    else:
    
        api_key = config["keys"][args.family]
        api = JUDGE_API_LIST[args.family]

        if not os.path.isdir(args.output_dir):
            os.makedirs(args.output_dir)

        responses = dict()

        if os.path.isdir(args.input_dir):
            for file in sorted(os.listdir(args.input_dir)):
                with open(os.path.join(args.input_dir, file), 'r') as fp:
                    response = json.load(fp)
                responses[response['query_id']] = response

        question = config["template_judge"]

        api(question, responses, args.model, api_key, args.output_dir)


if __name__ == "__main__":
    main()