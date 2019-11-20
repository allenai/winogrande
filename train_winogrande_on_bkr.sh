#!/bin/bash

set -e
set -x

export DESC_SUFFIX="suffix_for_experiment"
SHA=`git rev-parse HEAD`
export IMAGE_NAME="train-embs-${SHA}-${DESC_SUFFIX}"
EXPT_FILE=train_winogrande_on_bkr.yml

docker build -t ${IMAGE_NAME} .
export bp=$(beaker image create --name ${IMAGE_NAME} --quiet ${IMAGE_NAME})

echo ${bp}

export WINOGRANDE_DATASET=ds_xxxxxxxxxxxx # dataset on beaker

### parameters for experiment
export MODEL_TYPE=roberta_mc              # bert_mc | roberta_mc
export MODEL_NAME_OR_PATH=roberta-large   # bert-large-uncased | roberta-large  

export MOUNT_POINT=/data
export OUTPUT_DIR=/output/models/
export LOGGING_STEPS=4752
export SAVE_STEPS=4750
export SEED=$((RANDOM))
export MAX_SEQ_LENGTH=80
export DATA_DIR=${MOUNT_POINT}
export GPU_COUNT=1
export GROUP_ID=$((RANDOM))

TASKS=( winogrande )
BATCHSIZES=( 16 )
LEARNINGRATES=( 1e-5 )
EPOCHS=( 3 )
NUM_RANDOM_SEEDS=1


for (( c=1; c<=$NUM_RANDOM_SEEDS; c++ ))
do
  export SEED=$((RANDOM))
  for TASK in "${TASKS[@]}"
  do
    export TASK_NAME=${TASK}
    for s in "${BATCHSIZES[@]}"
    do
      export TRAIN_BATCH_SIZE=${s}
      export EVAL_BATCH_SIZE=${s}
      for l in "${LEARNINGRATES[@]}"
      do
        export LEARNING_RATE=${l}
        for e in "${EPOCHS[@]}"
        do
          export NUM_EPOCHS=${e}
          export WARMUP_PCT=0.1
          export DESC="$TASK_NAME-$GROUP_ID-$DESC_SUFFIX"
          expt=$(beaker experiment create --file ${EXPT_FILE})
        done
      done
    done
  done
done


echo "-->> Starting experiments with Group ID $GROUP_ID <<---"
