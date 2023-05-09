#!/bin/bash

if [ -z $1 ];
then echo "No CUDA device specified, exiting...";sleep 3;exit -1;
else CUDA_GPUS=$1;
fi

CUDA_NUM=`echo $CUDA_GPUS | awk -F, "{print NF}"`
CURRENT_DIR=`cd $(dirname $0);pwd`
# exit if code_dir is not defined
if [ -z $CODE_DIR ]
then echo 'env CODE_DIR is not defined, run source set_env.sh first!'; exit 1
fi

# indicate ITM task
read -p "Add ITM to training tasks? (y/N): " itm
case $itm in
    [yY])
        itm=true
        ;;
    [nN])
        itm=false
        ;;
    *)
        echo "Unknown input, defaulting to ITM=false..."
        ;;
esac

# file related consts
TAR_PREFIX="baseline"
if $itm; then TAR_PREFIX="${TAR_PREFIX}_itm"; fi

doc="$CURRENT_DIR/.max_cider_$TAR_PREFIX"
csv="$CURRENT_DIR/metrics_params/$TAR_PREFIX.csv"
conf=$CODE_DIR/conf/base_tokenized_pt.json
ckpt=$CODE_DIR/work_dir/pretrained/bare_model/output

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
    TAR_PREFIX=$TAR_PREFIX \
    CSV_FILENAME=$csv \
    CONF=$conf \
    CKPT_DIR=$ckpt \
    DISP_PARAM=false \
    ITM_TASK=$itm \
    ITM_WEIGHT=$itm_weight \
    bash $CURRENT_DIR/train_step1_base.sh $CUDA_GPUS
}

gen_current_best_from_csv

PARAM_CSV="$CURRENT_DIR/metrics_params/${TAR_PREFIX}_params.csv"

if [ -f $PARAM_CSV ];then
    for params in $(tail -n+2 "$PARAM_CSV")
    do
        if [ -z $params ]; then continue; fi
        # epoches,warm_up,lr,lr_end,weight_decay,batch_size
        IFS=',' read -r epoch warm_up lr lr_end weight_decay batch_size workers itm_weight <<< $params
        batch_size_eval=$((10#$batch_size*2))
        echo $epoch $warm_up $lr $lr_end $weight_decay $batch_size $workers

        # ITM Indicator
        printf "ITM Task Enabled: %s\n" $itm
        if $itm; then printf "ITM Task weight: %.1f\n" $itm_weight; fi
        train
    done
else
    epoch=5
    IFS=',' read warm_up lr lr_end weight_decay <<< "0.06,2e-05,7.5e-07,0.001"
    batch_size=12
    workers=8
    train
fi

cat $doc

