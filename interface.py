import argparse
import json
import os
from datasets import load_dataset

from call import API_LIST, get_logger, save_results, result_exists

def get_args(description='Benchly on LLM/VLMs'):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--llm', action='store_true', help="Whether to use text inputs only.")
    parser.add_argument("--model", type=str, default='gpt-4-0613', help='which model to use')
    parser.add_argument('--family', type=str, default='gpt', help='family for api key retrieval')
    parser.add_argument('--seed', type=int, default=42, help='Whether to evaluate random samples')
    parser.add_argument('--seed_size', type=int, default=5, help='Number of random samples')
    parser.add_argument('--diff_levels', type=list, default=[1, 3], help='difficulty levels to evaluate')

    parser.add_argument('--config', type=str, default='config.json', help='config file path')
    parser.add_argument('--output_dir', type=str, default='ckpts/', help='output directory')
    parser.add_argument('--data_path', type=str, default='data/datapath', help='dataset directory')

    args = parser.parse_args()

    return args

def main():
    args = get_args()

    with open(args.config, 'r') as fp:
        config = json.load(fp)

    api_key = config["keys"][args.family]
    api = API_LIST[args.family]
    
    output_path = args.output_dir + '/temp'

    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    
    logger = get_logger(os.path.join(args.output_dir, "log.txt"))

    if os.path.isfile(args.data_path):
        with open(args.data_path, 'r') as fp:
            dataset = json.load(fp)
    else:
        dataset = load_dataset(config["dataset"])["validation"].to_pandas()

        diff_levels_int = [int(level) for level in args.diff_levels if level != ',']
        dataset = dataset[dataset['difficulty_level'].isin(diff_levels_int)]
        
        if args.seed > 0:
            dataset = dataset.sample(n=args.seed_size, random_state=args.seed)

    logger.info("Experiment details:")
    logger.info('\t>>>seed: {}'.format(args.seed))
    logger.info('\t>>>random size: {}'.format(args.seed_size))
    logger.info('\t>>>model: {}'.format(args.model))
    logger.info('\t>>>text mode: {}'.format(args.llm))
    logger.info('\t>>>difficulty level: {}'.format(diff_levels_int))

    api(dataset, args.model, api_key, output_path, args.llm)

    responses = dict()

    for file in sorted(os.listdir(output_path)):
        with open(os.path.join(output_path, file), 'r') as fp:
            response = json.load(fp)
        responses[response['query_id']] = response

    save_path = os.path.join(args.output_dir, args.model + '.json')
    if not result_exists(save_path):
        save_results(save_path, responses)


if __name__ == "__main__":
    main()

