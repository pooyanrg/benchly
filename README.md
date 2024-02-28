# benchly

A simple benchmark api to prompt llm/vlms

## Files


1. `interface.py` - the interface to run api
2. `judge_interface.py` - the interface to run judge
3. `call.py` - requests/response engine

Usage:
```sh
python interface.py --llm --model "gpt-4-0413" --seed 42 --seed_size 5 --family "gpt" --config config.json --output_dir ckpts/
```
## Seed usage
set seed to 0 to evaluate the whole dataset.

Usage (judge):
```sh
python judge_interface.py --model "gpt-4-0413" --family "gpt" --config config.json --output_dir results/ --input_file ckpts/gemini-pro_response.json
```


## Config:
```
{"keys" : {"gemini" : ""},
 "dataset" : {"taesiri/simple_fsm_bench_long_text"},
  "template_judge":"Given the model the two strings, generate 0 if they are equivalent, otherwise generate 1.\n{model_output}\n{gt_answer}",
  }
```