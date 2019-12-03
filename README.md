# WinoGrande 

Version 1.1 (Dec 2nd, 2019)

- - - 

## Data

Download dataset by `download_winogrande.sh`

    ./data/
    ├── train_[xs,s,m,l,xl].jsonl          # training set with differnt sizes
    ├── train_[xs,s,m,l,xl]-labels.lst     # answer labels for training sets
    ├── dev.jsonl                          # development set
    ├── dev-labels.lst                     # answer labels for development set
    ├── test.jsonl                         # test set
    ├── sample-submissions-labels.lst      # example submission file for leaderboard    
    └── eval.py                            # evaluation script
    
You can use `train_*.jsonl` for training models and `dev` for validation.
Please note that labels are not included in `test.jsonl`. To evaluate your models on `test` set, make a submission to our [leaderboard](https://leaderboard.allenai.org/winogrande/submissions/public).


## Run experiments

### Setup

1. Download dataset by `download_winogrande.sh` 
1. `pip install -r requirements.txt`

### Training (fine-tuning)

1. You can train your model by `./scripts/run_experiment.py` (see `sample_training.sh`).

        e.g., 
        export PYTHONPATH=$PYTHONPATH:$(pwd)

        python scripts/run_experiment.py \
        --model_type roberta_mc \ 
        --model_name_or_path roberta-large \
        --task_name winogrande \
        --do_eval \
        --do_lower_case \
        --data_dir ./data \
        --max_seq_length 80 \
        --per_gpu_eval_batch_size 4 \
        --per_gpu_train_batch_size 16 \
        --learning_rate 1e-5 \
        --num_train_epochs 3 \
        --output_dir ./output/models/ \
        --do_train \
        --logging_steps 4752 \
        --save_steps 4750 \
        --seed 42 \
        --data_cache_dir ./output/cache/ \
        --warmup_pct 0.1 \
        --evaluate_during_training

1. If you have an access to [beaker](https://beaker.org/), you can run your experiments by `sh ./train_winogrande_on_bkr.sh`.

1. Results will be stored under `./output/models/`. 

### Prediction (on the test set)

1. You can make predictions by `./scripts/run_experiment.py` directly (see `sample_prediction.sh`).

        e.g., 
        export PYTHONPATH=$PYTHONPATH:$(pwd)

        python scripts/run_experiment.py \
        --model_type roberta_mc \
        --model_name_or_path .output/models \
        --task_name winogrande \
        --do_predict \
        --do_lower_case \
        --data_dir ./data \
        --max_seq_length 80 \
        --per_gpu_eval_batch_size 4 \
        --output_dir ./output/models/ \
        --data_cache_dir ./output/cache/ \

1. If you have an access to [beaker](https://beaker.org/), you can run your experiments  by `sh ./predict_winogrande_on_bkr.sh`.

1. Result is stored in `./output/models/predictions_test.lst`


## Evaluation

You can use `eval.py` for evaluation on the dev split, which yields `metrics.json`. 

    e.g., python eval.py --preds_file ./YOUR_PREDICTIONS.lst --labels_file ./dev-labels.lst

In the prediction file, each line consists of the predictions (1 or 2) by 5 training sets (ordered by `xs`, `s`, `m`, `l`, `xl`, separated by comma) for each evauation set question. 

     2,1,1,1,1
     1,1,2,2,2
     1,1,1,1,1
     .........
     .........

Namely, the first column is the predictions by a model trained/finetuned on `train_xs.jsonl`, followed by a model prediction by `train_s.jsonl`, ... , and the last (fifth) column is the predictions by a model from `train_xl.jsonl`.
Please checkout a sample submission file (`sample-submission-labels.lst`) for reference.


## Submission to Leaderboard

You can submit your predictions on `test` set to the [leaderboard](https://leaderboard.allenai.org/winogrande/submissions/public).
The submission file must be named as `predictions.lst`. The format is the same as above.  
    
## Reference
If you use this dataset, please cite the following paper:

	@article{sakaguchi2019winogrande,
	    title={WinoGrande: An Adversarial Winograd Schema Challenge at Scale},
	    author={Sakaguchi, Keisuke and Bras, Ronan Le and Bhagavatula, Chandra and Choi, Yejin},
	    journal={arXiv preprint arXiv:1907.10641},
	    year={2019}
	}


## License 

Winogrande (codebase) is licensed under the Apache License 2.0. The dataset is licensed under CC-BY.


## Questions?

Please file GitHub issues with your questions/suggestions. You may also ask us questions at our [google group](https://groups.google.com/a/allenai.org/forum/#!forum/winogrande).


## Contact 

Email: keisukes[at]allenai.org
