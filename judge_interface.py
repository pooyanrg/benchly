import argparse
import json
import os

import logging

from call import JUDGE_API_LIST, get_logger, save_results

def get_args(description='Benchly Judge Evaluation'):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--model", type=str, default='gpt-4-0613', help='which model to use')
    parser.add_argument('--family', type=str, default='gpt', help='family for api key retrieval')

    parser.add_argument('--config', type=str, default='config.json', help='config file path')
    parser.add_argument('--input_file', type=str, default='ckpts/gemini-proresponse.json', help='input directory')
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
            os.makedirs(args.output_dir + '/temp/')
        
        logger = get_logger(os.path.join(args.output_dir, "log.txt"))

        with open(args.input_file, 'r') as fp:
            responses = json.load(fp)

        question = config["template_judge"]

        logger.info("Experiment details:")
        logger.info('\t>>>model: {}'.format(args.model))
        logger.info('\t>>>query: {}'.format(question))

        api(question, responses, args.model, api_key, args.output_dir + '/temp/')

        answers = dict()
        for file in sorted(os.listdir(args.output_dir + '/temp/')):
            with open(os.path.join(args.output_dir + '/temp/', file), 'r') as fp:
                answer = json.load(fp)
            answers[answer['query_id']] = answer

        save_results(args.output_dir, args.model, answers)


if __name__ == "__main__":
    main()