# benchly

A simple benchmark api to prompt llm/vlms

## Files


1. `interface.py` - the interface to run api
2. `judge_interface.py` - the interface to run judge
3. `call.py` - requests/response engine

## Config:
```
{"keys" : {"gemini" : ""},
 "dataset" : {"taesiri/simple_fsm_bench_long_text"},
  "template_judge":"Given the model the two strings, generate 0 if they are equivalent, otherwise generate 1.\n{model_output}\n{gt_answer}",
  }
```

## Prompting

Usage:
```sh
python interface.py --llm --model "gpt-4-0413" --seed 42 --seed_size 5 --family "gpt" --config config.json --output_dir ckpts/
```
# Seed usage
set seed to 0 to evaluate the whole dataset.

# output

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

# output

for each query the script will generate a json file: 

```
{"query_id" : ,
 "query" : ,
  "judge_response":,
  "gt_answer":,
  }
```