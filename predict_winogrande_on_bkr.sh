#!/bin/bash

set -e
set -x

export DESC_SUFFIX="suffix_for_experiment"
SHA=`git rev-parse HEAD`
export IMAGE_NAME="pred-embs-${SHA}-${DESC_SUFFIX}"
EXPT_FILE=predict_winogrande_on_bkr.yml

docker build -t ${IMAGE_NAME} .
export bp=$(beaker image create --name ${IMAGE_NAME} --quiet ${IMAGE_NAME})

echo ${bp}

export WINOGRANDE_DATASET=ds_xxxxxxxxxxxx # dataset on beaker
export MODEL_DATASET=ds_xxxxxxxxxxxx      # trained model on beaker

### parameters for experiment
export MODEL_TYPE=roberta_mc              # bert_mc | roberta_mc
export MODEL_NAME_OR_PATH="/model/models"
export MOUNT_POINT=/data
export MODEL_MOUNT_POINT="/model"
export OUTPUT_DIR=/output/models/
export MAX_SEQ_LENGTH=80
export DATA_DIR=${MOUNT_POINT}
export GPU_COUNT=1
export GROUP_ID=$((RANDOM))

TASKS=( winogrande )
BATCHSIZES=( 4 )

for TASK in "${TASKS[@]}"
do
  export TASK_NAME=${TASK}
  for s in "${BATCHSIZES[@]}"
  do
    export EVAL_BATCH_SIZE=${s}
    export DESC="$TASK_NAME-$GROUP_ID-$DESC_SUFFIX"
    expt=$(beaker experiment create --file ${EXPT_FILE})
  done
done


echo "-->> Starting experiments with Group ID $GROUP_ID <<---"
