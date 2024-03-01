# benchly

A simple benchmark api to prompt llm/vlms

## Files

1. `run_benchly.sh` - script to run the pipeline
2. `interface.py` - the interface to run api
3. `judge_interface.py` - the interface to run judge
4. `call.py` - requests/response engine

## Config:
```
{"keys" : {"gemini" : ""},
    "dataset" : {"taesiri/simple_fsm_bench_long_text"},
    "experiment": "transition_matrix_text_only"
    "template_judge":"Given the model the two strings, generate 0 if they are equivalent, otherwise generate 1.\n{model_output}\n{gt_answer}",
  }
```
## How to use the pipeline
```sh
bash run_benchly.sh
```


## Prompting

Usage:
```sh
python interface.py --llm --model "gpt-4-0413" --seed 42 --seed_size 5 --family "gpt" --config config.json --output_dir ckpts/
```
Seed usage:
set seed to 0 to evaluate the whole dataset.

Output:

for each query the script will generate a json file: 

```
{"query_id" : ,
 "query" : ,
  "response":,
  "gt_answer":,
  }
```

## Judge

Usage:
```sh
python judge_interface.py --model "gpt-4-0413" --family "gpt" --config config.json --output_dir results/ --input_file ckpts/gemini-pro_response.json
```

Output

for each query the script will generate a json file: 

```
{"query_id" : ,
 "query" : ,
  "judge_response":,
  "gt_answer":,
  }
```