#!/bin/bash

if [ -z $1 ];
then echo "No CUDA device specified, exiting...";sleep 3;exit -1;
else CUDA_GPUS=$1;
fi

CUDA_NUM=`echo $CUDA_GPUS | awk -F, "{print NF}"`
CURRENT_DIR=`cd $(dirname $0);pwd`
[ -z $CODE_DIR ] && (echo '$CODE_DIR is not defined, run source set_env.sh first!'; exit 1)
# file related consts
TAR_PREFIX=base_pt
doc="$CURRENT_DIR/.max_cider_$TAR_PREFIX"
csv="$CURRENT_DIR/metrics_params/$TAR_PREFIX.csv"
conf=$CODE_DIR/conf/scst_test/base_lr1e-5.json
ckpt=$CODE_DIR/work_dir/pretrained/base_pt

function get_cider_column()
{
    header=$(head -n1 $csv)
    header_no=$(echo $header | awk -F, '{print NF}')
    # iterate through the header
    for ((i=1; i<=$header_no; i++))
    do
        if [ $(echo $header | cut -d, -f$i | tr A-Z a-z) == "cider" ]
        then echo $i; return 0;
        fi
    done
    echo -1
    return -1
}

function gen_current_best_from_csv()
{
    # if exists then delete it
    [ -f $doc ] && rm $doc
    CIDER_COL=$(get_cider_column)
    # we default that a base data exists here
    # <<< means inputing the result as a string
    (awk -F, "NR>1{a[\$1]+=\$($CIDER_COL)}END{for(i in a) printf \"%s,%.2f\n\", i, a[i]}" $csv | sort -nr -k2 | head -n1 ) > $doc
}

# train
function train()
{
    time -p \
    LR=$lr \
    LR_END=$lr_end \
    WARM_UP=$warm_up \
    W_DECAY=$weight_decay \
    EPOCH=$epoch \
    BATCH=$batch_size \
    BATCH_EVAL=$batch_size_eval \
    WORKERS=$workers \
    DOC_FILEDIR=$doc \
    TAR_PREFIX=$TAR_PREFIX \
    CSV_FILENAME=$csv \
    CONF=$conf \
    CKPT_DIR=$ckpt \
    FREEZE_RES=$freeze_resnet \
    DISP_PARAM=false \
    bash $CURRENT_DIR/train_step2_base.sh $CUDA_GPUS
}

gen_current_best_from_csv

PARAM_CSV="$CURRENT_DIR/metrics_params/${TAR_PREFIX}_params.csv"

if [ -f $PARAM_CSV ]
then
    for params in $(tail -n+2 "$PARAM_CSV")
    do
        if [ -z $params ]; then continue; fi
        # epoches,warm_up,lr,lr_end,weight_decay,batch_size
        IFS=',' read -r epoch warm_up lr lr_end weight_decay batch_size freeze_resnet <<< $params
        workers=4
        echo $epoch $warm_up $lr $lr_end $weight_decay $batch_size $freeze_resnet
        train
    done
else
    epoch=3
    IFS=',' read warm_up lr lr_end weight_decay <<< "0.06,2e-05,7.5e-07,0.001"
    batch_size=24
    batch_size_eval=32
    workers=8
    freeze_resnet=false
    train
fi

cat $doc

