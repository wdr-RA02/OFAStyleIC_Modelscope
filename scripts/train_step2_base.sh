#!/bin/bash

# train CIDEr for a single turn
# as the function body of the outer loop @ train_step2_mul_base.sh

# define a fn to implement getattr() like that of python

if [ -z $1 ];
then echo "No CUDA device specified, exiting...";sleep 3;exit -1;
else CUDA_GPUS=$1;
fi

CURRENT_DIR=$(cd $(dirname $0);pwd)
CUDA_NUM=$(echo $CUDA_GPUS | awk -F, "{print NF}")
TAR_PREFIX=${TAR_PREFIX:-"base_pt"}
# exit if code_dir is not defined
if [ -z $CODE_DIR ]
then echo 'env CODE_DIR is not defined, run source set_env.sh first!'; exit 1
fi
MODEL_OP=$CODE_DIR/model_operator.py
DISP_PARAM=${DISP_PARAM:-true}

CONF=${CONF:-$CODE_DIR/conf/scst_test/base_lr1e-5.json}
CKPT_DIR=${CKPT_DIR:-$CODE_DIR/work_dir/pretrained/base_pt}
WORK_DIR=$(jq ."work_dir" $CONF | sed -e "s#.*/.*#\0#g" -e "s#\.#$CODE_DIR#g" -e "s#\"##g")
# exit if conf is not correctly read
if [ $? != 0 ];then exit $?;fi
WORK_DIR=${WORK_DIR::-1}
SAVE_DIR=$WORK_DIR/saved_models

CSV_FILENAME=${CSV_FILENAME:-"$CURRENT_DIR/metrics_params/$TAR_PREFIX.csv"}
DOC_FILEDIR=${DOC_FILEDIR:-$CURRENT_DIR/.max_cider_$TAR_PREFIX}

[ -d $SAVE_DIR ] || mkdir -p $SAVE_DIR
DDP_PORT=30002

# chdir to CODE_DIR to ensure conf_file works
cd $CODE_DIR
echo $WORK_DIR

# that's how getattr() is done
lr=${LR:-1e-5}
lr_end=${LR_END:-1e-7}
warm_up=${WARM_UP:-0.02}
weight_decay=${W_DECAY:-0.001}
epoch=${EPOCH:-3}
batch_size=${BATCH:-24}
batch_size_eval=${BATCH_EVAL:-56}
workers=${WORKERS:-8}
FREEZE_RES=${FREEZE_RES:-true}

if $DISP_PARAM
then
    for x in $lr $lr_end $warm_up $weight_decay;
    do echo $x;
    done
fi
# train
echo $CUDA_GPUS
SCST_PROMPT="torchrun --rdzv_backend c10d \
        --rdzv_endpoint localhost:$DDP_PORT \
        --nnodes 1 --nproc_per_node $CUDA_NUM \
        $MODEL_OP train --conf $CONF \
        --checkpoint $CKPT_DIR \
        --cider \
        --lr $lr \
        --lr_end $lr_end\
        --warm_up $warm_up \
        --weight_decay $weight_decay \
        --max_epoches $epoch \
        --batch_size $batch_size \
        --num_workers $workers"

if $FREEZE_RES; then
    CUDA_VISIBLE_DEVICES=$CUDA_GPUS $SCST_PROMPT --freeze_resnet
else
    CUDA_VISIBLE_DEVICES=$CUDA_GPUS $SCST_PROMPT
fi

if [ $? != 0 ]; then exit $?; fi;

# eval
source $CURRENT_DIR/eval_model.sh `echo $CUDA_GPUS | cut -d, -f1`
## save to $SAVE_DIR/base_pt_{dt}.tar.gz

function get_max_cider()
{
    if [ ! -f $DOC_FILEDIR ]
    then last_max_cider=0
    else
        last_dt=$(awk -F, '{print $1}' $DOC_FILEDIR)
        last_max_cider=$(awk -F, '{print $2}' $DOC_FILEDIR)
    fi
}

function get_cider_column() {
    local header=$(head -n1 "$CSV_FILENAME")
    local cider_index=$(echo "$header" | awk -F, '{ for(i=1; i<=NF; i++) if(tolower($i)=="cider") print i }')
    if [ -z "$cider_index" ]; then
        echo "Error: column 'cider' not found in file $CSV_FILENAME" >&2
        return 1
    else
        echo "$cider_index"
        return 0
    fi
}

# get the last highest date_id
get_max_cider
# it indicates which column the CIDEr score is stored in CSV
CIDER_COL=$(get_cider_column)
# obtain the cider score just now
read eval_dt cider_score <<< $(tail -n1 $CSV_FILENAME | awk -F, "{print \$1,\$($CIDER_COL)}")
# archive if the CIDEr point is greater
ARCHIVE_THIS=$(echo "$cider_score >= $last_max_cider" | bc)

if [ $ARCHIVE_THIS == 1 ] 
then
    echo "Saving new best CIDEr model to $SAVE_DIR/$TAR_PREFIX-$eval_dt.tar.gz"
    PAST_DIR=$(pwd)
    cd $WORK_DIR/output
    tar czf - ./pytorch_model.bin | pv > $SAVE_DIR/$TAR_PREFIX-$eval_dt.tar.gz
    # write current record to the file
    cd $PAST_DIR
    echo "$eval_dt,$cider_score" > $DOC_FILEDIR
fi

lsof -i:$DDP_PORT