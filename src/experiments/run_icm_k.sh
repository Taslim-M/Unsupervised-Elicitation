#!/bin/bash
set -e

python ICM.py \
    --testbed truthfulQA \
    --alpha 70 \
    --K 500 \
    --model meta-llama/Llama-3.1-8B \
    --num_seed 4 \
    --batch_size 12 \
    --file_name Brazil_12_part3.json \
    --continue_from_existing 1

python ICM.py \
    --testbed truthfulQA \
    --alpha 70 \
    --K 500 \
    --model meta-llama/Llama-3.1-8B \
    --num_seed 8 \
    --batch_size 14 \
    --file_name Brazil_14_part1.json \
    --continue_from_existing 1

python ICM.py \
    --testbed truthfulQA \
    --alpha 70 \
    --K 500 \
    --model meta-llama/Llama-3.1-8B \
    --num_seed 8 \
    --batch_size 18 \
    --file_name Brazil_18_part4.json \
    --continue_from_existing 1

python ICM.py \
    --testbed truthfulQA \
    --alpha 40 \
    --K 500 \
    --model meta-llama/Llama-3.1-8B \
    --num_seed 4 \
    --batch_size 20 \
    --file_name Brazil_20_part2.json \
    --continue_from_existing 1

python ICM.py \
    --testbed truthfulQA \
    --alpha 50 \
    --K 500 \
    --model meta-llama/Llama-3.1-8B \
    --num_seed 8 \
    --batch_size 28 \
    --file_name Britain_28_part1.json \
    --continue_from_existing 1

python ICM.py \
    --testbed truthfulQA \
    --alpha 40 \
    --K 500 \
    --model meta-llama/Llama-3.1-8B \
    --num_seed 4 \
    --batch_size 28 \
    --file_name Britain_28_part2.json \
    --continue_from_existing 1

python ICM.py \
    --testbed truthfulQA \
    --alpha 70 \
    --K 500 \
    --model meta-llama/Llama-3.1-8B \
    --num_seed 8 \
    --batch_size 28 \
    --file_name Britain_28_part3.json \
    --continue_from_existing 1

python ICM.py \
    --testbed truthfulQA \
    --alpha 40 \
    --K 500 \
    --model meta-llama/Llama-3.1-8B \
    --num_seed 8 \
    --batch_size 28 \
    --file_name Britain_28_part4.json \
    --continue_from_existing 1

python ICM.py \
    --testbed truthfulQA \
    --alpha 50 \
    --K 500 \
    --model meta-llama/Llama-3.1-8B \
    --num_seed 4 \
    --batch_size 28 \
    --file_name France_28_part1.json \
    --continue_from_existing 1

python ICM.py \
    --testbed truthfulQA \
    --alpha 50 \
    --K 500 \
    --model meta-llama/Llama-3.1-8B \
    --num_seed 8 \
    --batch_size 28 \
    --file_name France_28_part2.json \
    --continue_from_existing 1

python ICM.py \
    --testbed truthfulQA \
    --alpha 50 \
    --K 500 \
    --model meta-llama/Llama-3.1-8B \
    --num_seed 8 \
    --batch_size 28 \
    --file_name France_28_part3.json \
    --continue_from_existing 1

python ICM.py \
    --testbed truthfulQA \
    --alpha 40 \
    --K 500 \
    --model meta-llama/Llama-3.1-8B \
    --num_seed 8 \
    --batch_size 28 \
    --file_name France_28_part4.json \
    --continue_from_existing 1

python ICM.py \
    --testbed truthfulQA \
    --alpha 40 \
    --K 500 \
    --model meta-llama/Llama-3.1-8B \
    --num_seed 8 \
    --batch_size 28 \
    --file_name Germany_28_part1.json \
    --continue_from_existing 1

python ICM.py \
    --testbed truthfulQA \
    --alpha 50 \
    --K 500 \
    --model meta-llama/Llama-3.1-8B \
    --num_seed 8 \
    --batch_size 28 \
    --file_name Germany_28_part2.json \
    --continue_from_existing 1

python ICM.py \
    --testbed truthfulQA \
    --alpha 40 \
    --K 500 \
    --model meta-llama/Llama-3.1-8B \
    --num_seed 8 \
    --batch_size 28 \
    --file_name Germany_28_part3.json \
    --continue_from_existing 1

python ICM.py \
    --testbed truthfulQA \
    --alpha 40 \
    --K 500 \
    --model meta-llama/Llama-3.1-8B \
    --num_seed 8 \
    --batch_size 28 \
    --file_name Germany_28_part4.json \
    --continue_from_existing 1

python ICM.py \
    --testbed truthfulQA \
    --alpha 70 \
    --K 500 \
    --model meta-llama/Llama-3.1-8B \
    --num_seed 4 \
    --batch_size 18 \
    --file_name Indonesia_18_part3.json \
    --continue_from_existing 1

python ICM.py \
    --testbed truthfulQA \
    --alpha 70 \
    --K 500 \
    --model meta-llama/Llama-3.1-8B \
    --num_seed 4 \
    --batch_size 22 \
    --file_name Indonesia_22_part1.json \
    --continue_from_existing 1

python ICM.py \
    --testbed truthfulQA \
    --alpha 40 \
    --K 500 \
    --model meta-llama/Llama-3.1-8B \
    --num_seed 8 \
    --batch_size 24 \
    --file_name Indonesia_24_part4.json \
    --continue_from_existing 1

python ICM.py \
    --testbed truthfulQA \
    --alpha 50 \
    --K 500 \
    --model meta-llama/Llama-3.1-8B \
    --num_seed 8 \
    --batch_size 28 \
    --file_name Indonesia_28_part2.json \
    --continue_from_existing 1

python ICM.py \
    --testbed truthfulQA \
    --alpha 50 \
    --K 500 \
    --model meta-llama/Llama-3.1-8B \
    --num_seed 4 \
    --batch_size 28 \
    --file_name Japan_28_part1.json \
    --continue_from_existing 1

python ICM.py \
    --testbed truthfulQA \
    --alpha 40 \
    --K 500 \
    --model meta-llama/Llama-3.1-8B \
    --num_seed 8 \
    --batch_size 28 \
    --file_name Japan_28_part2.json \
    --continue_from_existing 1

python ICM.py \
    --testbed truthfulQA \
    --alpha 50 \
    --K 500 \
    --model meta-llama/Llama-3.1-8B \
    --num_seed 8 \
    --batch_size 28 \
    --file_name Japan_28_part3.json \
    --continue_from_existing 1

python ICM.py \
    --testbed truthfulQA \
    --alpha 50 \
    --K 500 \
    --model meta-llama/Llama-3.1-8B \
    --num_seed 8 \
    --batch_size 28 \
    --file_name Japan_28_part4.json \
    --continue_from_existing 1

python ICM.py \
    --testbed truthfulQA \
    --alpha 40 \
    --K 500 \
    --model meta-llama/Llama-3.1-8B \
    --num_seed 8 \
    --batch_size 28 \
    --file_name Jordan_28_part1.json \
    --continue_from_existing 1

python ICM.py \
    --testbed truthfulQA \
    --alpha 40 \
    --K 500 \
    --model meta-llama/Llama-3.1-8B \
    --num_seed 8 \
    --batch_size 28 \
    --file_name Jordan_28_part2.json \
    --continue_from_existing 1

python ICM.py \
    --testbed truthfulQA \
    --alpha 50 \
    --K 500 \
    --model meta-llama/Llama-3.1-8B \
    --num_seed 8 \
    --batch_size 28 \
    --file_name Jordan_28_part3.json \
    --continue_from_existing 1

python ICM.py \
    --testbed truthfulQA \
    --alpha 50 \
    --K 500 \
    --model meta-llama/Llama-3.1-8B \
    --num_seed 8 \
    --batch_size 28 \
    --file_name Jordan_28_part4.json \
    --continue_from_existing 1

python ICM.py \
    --testbed truthfulQA \
    --alpha 40 \
    --K 500 \
    --model meta-llama/Llama-3.1-8B \
    --num_seed 8 \
    --batch_size 28 \
    --file_name Lebanon_28_part1.json \
    --continue_from_existing 1

python ICM.py \
    --testbed truthfulQA \
    --alpha 50 \
    --K 500 \
    --model meta-llama/Llama-3.1-8B \
    --num_seed 8 \
    --batch_size 28 \
    --file_name Lebanon_28_part2.json \
    --continue_from_existing 1

python ICM.py \
    --testbed truthfulQA \
    --alpha 50 \
    --K 500 \
    --model meta-llama/Llama-3.1-8B \
    --num_seed 8 \
    --batch_size 28 \
    --file_name Lebanon_28_part3.json \
    --continue_from_existing 1

python ICM.py \
    --testbed truthfulQA \
    --alpha 40 \
    --K 500 \
    --model meta-llama/Llama-3.1-8B \
    --num_seed 4 \
    --batch_size 28 \
    --file_name Lebanon_28_part4.json \
    --continue_from_existing 1

python ICM.py \
    --testbed truthfulQA \
    --alpha 40 \
    --K 500 \
    --model meta-llama/Llama-3.1-8B \
    --num_seed 4 \
    --batch_size 28 \
    --file_name Mexico_28_part1.json \
    --continue_from_existing 1

python ICM.py \
    --testbed truthfulQA \
    --alpha 50 \
    --K 500 \
    --model meta-llama/Llama-3.1-8B \
    --num_seed 8 \
    --batch_size 28 \
    --file_name Mexico_28_part2.json \
    --continue_from_existing 1

python ICM.py \
    --testbed truthfulQA \
    --alpha 50 \
    --K 500 \
    --model meta-llama/Llama-3.1-8B \
    --num_seed 8 \
    --batch_size 28 \
    --file_name Mexico_28_part3.json \
    --continue_from_existing 1

python ICM.py \
    --testbed truthfulQA \
    --alpha 40 \
    --K 500 \
    --model meta-llama/Llama-3.1-8B \
    --num_seed 4 \
    --batch_size 28 \
    --file_name Mexico_28_part4.json \
    --continue_from_existing 1

python ICM.py \
    --testbed truthfulQA \
    --alpha 50 \
    --K 500 \
    --model meta-llama/Llama-3.1-8B \
    --num_seed 8 \
    --batch_size 12 \
    --file_name Nigeria_12_part3.json \
    --continue_from_existing 1

python ICM.py \
    --testbed truthfulQA \
    --alpha 70 \
    --K 500 \
    --model meta-llama/Llama-3.1-8B \
    --num_seed 4 \
    --batch_size 14 \
    --file_name Nigeria_14_part1.json \
    --continue_from_existing 1

python ICM.py \
    --testbed truthfulQA \
    --alpha 70 \
    --K 500 \
    --model meta-llama/Llama-3.1-8B \
    --num_seed 8 \
    --batch_size 16 \
    --file_name Nigeria_16_part2.json \
    --continue_from_existing 1

python ICM.py \
    --testbed truthfulQA \
    --alpha 50 \
    --K 500 \
    --model meta-llama/Llama-3.1-8B \
    --num_seed 4 \
    --batch_size 18 \
    --file_name Nigeria_18_part4.json \
    --continue_from_existing 1

python ICM.py \
    --testbed truthfulQA \
    --alpha 50 \
    --K 500 \
    --model meta-llama/Llama-3.1-8B \
    --num_seed 8 \
    --batch_size 28 \
    --file_name Pakistan_28_part1.json \
    --continue_from_existing 1

python ICM.py \
    --testbed truthfulQA \
    --alpha 70 \
    --K 500 \
    --model meta-llama/Llama-3.1-8B \
    --num_seed 8 \
    --batch_size 28 \
    --file_name Pakistan_28_part2.json \
    --continue_from_existing 1

python ICM.py \
    --testbed truthfulQA \
    --alpha 70 \
    --K 500 \
    --model meta-llama/Llama-3.1-8B \
    --num_seed 8 \
    --batch_size 28 \
    --file_name Pakistan_28_part3.json \
    --continue_from_existing 1

python ICM.py \
    --testbed truthfulQA \
    --alpha 50 \
    --K 500 \
    --model meta-llama/Llama-3.1-8B \
    --num_seed 8 \
    --batch_size 28 \
    --file_name Pakistan_28_part4.json \
    --continue_from_existing 1

python ICM.py \
    --testbed truthfulQA \
    --alpha 70 \
    --K 500 \
    --model meta-llama/Llama-3.1-8B \
    --num_seed 8 \
    --batch_size 28 \
    --file_name Russia_28_part1.json \
    --continue_from_existing 1

python ICM.py \
    --testbed truthfulQA \
    --alpha 40 \
    --K 500 \
    --model meta-llama/Llama-3.1-8B \
    --num_seed 8 \
    --batch_size 28 \
    --file_name Russia_28_part2.json \
    --continue_from_existing 1

python ICM.py \
    --testbed truthfulQA \
    --alpha 40 \
    --K 500 \
    --model meta-llama/Llama-3.1-8B \
    --num_seed 8 \
    --batch_size 28 \
    --file_name Russia_28_part3.json \
    --continue_from_existing 1

python ICM.py \
    --testbed truthfulQA \
    --alpha 50 \
    --K 500 \
    --model meta-llama/Llama-3.1-8B \
    --num_seed 8 \
    --batch_size 28 \
    --file_name Russia_28_part4.json \
    --continue_from_existing 1

python ICM.py \
    --testbed truthfulQA \
    --alpha 70 \
    --K 500 \
    --model meta-llama/Llama-3.1-8B \
    --num_seed 8 \
    --batch_size 28 \
    --file_name Turkey_28_part1.json \
    --continue_from_existing 1

python ICM.py \
    --testbed truthfulQA \
    --alpha 40 \
    --K 500 \
    --model meta-llama/Llama-3.1-8B \
    --num_seed 8 \
    --batch_size 28 \
    --file_name Turkey_28_part2.json \
    --continue_from_existing 1

python ICM.py \
    --testbed truthfulQA \
    --alpha 40 \
    --K 500 \
    --model meta-llama/Llama-3.1-8B \
    --num_seed 8 \
    --batch_size 28 \
    --file_name Turkey_28_part3.json \
    --continue_from_existing 1

python ICM.py \
    --testbed truthfulQA \
    --alpha 70 \
    --K 500 \
    --model meta-llama/Llama-3.1-8B \
    --num_seed 8 \
    --batch_size 28 \
    --file_name Turkey_28_part4.json \
    --continue_from_existing 1
