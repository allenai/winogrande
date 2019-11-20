#!/bin/sh

CUDA_VISIBLE_DEVICES=0 python ./scripts/run_experiment.py \
--model_type roberta_mc \
--model_name_or_path ./output/models \
--task_name winogrande \
--do_predict \
--do_lower_case \
--data_dir ./data \
--max_seq_length 80 \
--per_gpu_eval_batch_size 4 \
--output_dir ./output/models/ \
--data_cache_dir ./output/cache/ \
