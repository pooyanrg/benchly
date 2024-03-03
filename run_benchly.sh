#!/bin/bash

DIR=experiment/
mkdir ${DIR}

python interface.py --llm \
--model "gemini-pro" \
--seed 42 \
--seed_size 1000 \
--family "gemini" \
--config config.json \
--output_dir ${DIR}ckpts/

python judge_interface.py \
--model "gemini-pro" \
--family "gemini" \
--config config.json \
--output_dir ${DIR}results/ \
--input_file ${DIR}ckpts/gemini-pro.json