export GEMINI_API_KEY="your_api_key"
export OPENAI_API_KEY="your_api_key"
export COHERE_API_KEY="your_api_key"
export HUGGINGFACE_API_KEY="your_api_key"
export TOGETHERAI_API_KEY="your_api_key"


python lite_api.py --llm --model "gpt-3.5-turbo" --family "gpt" --seed 42 --seed_size 1000 --num_retries 10 --diff_levels [1, 2, 3] --config config.json --experiment ckpts/