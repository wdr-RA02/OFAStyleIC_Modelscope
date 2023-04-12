#!/bin/bash
if [ -z $1 ] && [ -z $CUDA_VISIBLE_DEVICES ]
then echo "No CUDA device specified, exiting..." >&2;sleep 3;exit -1;
else CUDA_GPUS=$(echo $1|cut -d, -f1);
fi

CUDA_NUM=$(echo $CUDA_GPUS | awk -F, "{print NF}")

if [ -z $CONF ]
then echo "CONF variable is empty, exiting..." >&2 ;sleep 3;exit -2;
fi

CURRENT_DIR=$(cd $(dirname $0);pwd)
[ -z $CODE_DIR ] && (echo '$CODE_DIR is not defined, run source set_env.sh first!'; exit 1)
MODEL_OP=$CODE_DIR/model_operator.py
DDP_PORT=30010
# chdir to CODE_DIR to ensure conf_file works
cd $CODE_DIR

batch_size_eval=${BATCH_EVAL:-32}
workers=${WORKERS:-8}


function evaluate_model(){
    base_command="torchrun --rdzv_backend c10d \
        --rdzv_endpoint localhost:$DDP_PORT \
        --nnodes 1 --nproc_per_node 1 \
        $MODEL_OP eval --conf $CONF \
        --batch_size $batch_size_eval \
        --num_workers $workers \
        --patch_image_size 256"
        
    if [ $CSV_FILENAME ]
    then
        base_command="$base_command --log_csv_file $CSV_FILENAME"
    fi

    $base_command
}

echo $CSV_FILENAME

CUDA_VISIBLE_DEVICES=$CUDA_GPUS evaluate_model

