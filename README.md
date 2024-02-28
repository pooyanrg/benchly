# benchly

A simple benchmark api to prompt llm/vlms

## Files


1. `interface.py` - the interface to run api
2. `call.py` - requests/response engine

Usage:
```sh
python interface.py --llm --model --seed 42 --seed_size 5 "gpt-4-0413" --family "gpt" --config config.json --output_dir ckpts/
```
#seed usage
set seed to 0 to evaluate the whole dataset.



##Config:
```
{"keys" : {"gemini" : ""}, "dataset" : {"taesiri/simple_fsm_bench_long_text"}}
```