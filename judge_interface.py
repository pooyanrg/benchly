import argparse
import json
import os

import logging

from call import JUDGE_API_LIST, get_logger, save_results, result_exists

def get_args(description='Benchly Judge Evaluation'):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--model", type=str, default='gpt-4-0613', help='which model to use')
    parser.add_argument('--family', type=str, default='gpt', help='family for api key retrieval')

    parser.add_argument('--config', type=str, default='config.json', help='config file path')
    parser.add_argument('--input_file', type=str, default='ckpts/gemini-pro.json', help='input directory')
    parser.add_argument('--output_dir', type=str, default='results/', help='output directory')

    args = parser.parse_args()

    return args

def main():
    args = get_args()

    with open(args.config, 'r') as fp:
        config = json.load(fp)

    api_key = config["keys"][args.family]
    api = JUDGE_API_LIST[args.family]

    output_path = args.output_dir + '/temp'
    
    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    
    logger = get_logger(os.path.join(args.output_dir, "log.txt"))

    with open(args.input_file, 'r') as fp:
        responses = json.load(fp)

    question = config["template_judge"]

    logger.info("Experiment details:")
    logger.info('\t>>>model: {}'.format(args.model))
    logger.info('\t>>>query: {}'.format(question))

    api(question, responses, args.model, api_key, output_path)

    answers = dict()
    for file in sorted(os.listdir(output_path)):
        with open(os.path.join(output_path, file), 'r') as fp:
            answer = json.load(fp)
        answers[answer['query_id']] = answer

    save_path = os.path.join(args.output_dir, args.model + '.json')
    if not result_exists(save_path):
        save_results(save_path, answers)


if __name__ == "__main__":
    main()