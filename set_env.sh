#!/bin/bash

# activate the env
source $HOME/anaconda3/bin/activate modelscope_py38_cu102

# custom config just for ease
alias ps_me="ps -o euser=EUSER_____________,pid,cmd=cmd__________________________________,etime -u $USER"
tasklist_pid(){
    # merge all pids into a regex
    for i in $@
    do
        pids="${pids}${i}|"
    done
    pids=${pids:0:-1}
    ps -eo "euser=EUSER_____________,pid,cmd=cmd_________________,etime" | grep -v grep | grep -E "^EUSER|$pids"
}
alias nvsmi_freemem="nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits"
watch_nvsmi(){
    if [ -z $2 ];then time_s=0.5;else time_s=$2;fi
    if [ -z $1 ];then cards="";else cards=" -i $1";fi
    watch -n $time_s "nvidia-smi"$cards
}

watch_nvsmi_simple(){
    if [ -z $2 ];then time_s=0.5;else time_s=$2;fi
    if [ -z $1 ];then cards="";else cards="-i $1";fi
    watch -n $time_s "nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader $cards"
}

# useful aliases
alias xe_base_mul="./work_dir/scripts/train_step1_mul_base.sh"
alias cider_base_mul="./work_dir/scripts/train_step2_mul_base.sh"
alias inference="python3 model_operator.py inference"
alias watch_baseline_csv="./work_dir/scripts/watch_csv.sh 0.5 ./work_dir/scripts/metrics_params/baseline.csv"
alias watch_cider_csv="./work_dir/scripts/watch_csv.sh 0.5 ./work_dir/scripts/metrics_params/base_pt.csv"

# export these to other shells
export -f tasklist_pid
export -f watch_nvsmi
export -f watch_nvsmi_simple

