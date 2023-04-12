#!/bin/bash

# activate the env
if [ -z $(which conda) ]
then
    echo "conda not exist in your system. "
    return
fi

export CODE_DIR=$HOME/codes/OFAStyle
echo "Set code dir to $CODE_DIR"

function set_anaconda_env()
{
    env_lists=$(conda env list | tail -n+3 | awk '{print $1}')
    PS3="Please specify an environment to activate, Ctrl-C to exit: "
    select env in ${env_lists[@]}
    do
        if [ -z $env ]
        then 
            echo "Invalid option. "
        else
            conda activate $env
            echo "Activated environment $env"
            break
        fi
    done
    rm -rv $CODE_DIR/workspace/*
}

[ "$1" != "skip" ] && set_anaconda_env || echo "Skipped setting up anaconda environment"
# specify an environment to activate

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
alias xe_base_mul="$CODE_DIR/scripts/train_step1_mul_base.sh"
alias cider_base_mul="$CODE_DIR/scripts/train_step2_mul_base.sh"
alias inference="python3 model_operator.py inference"
alias watch_baseline_csv="$CODE_DIR/scripts/watch_csv.sh 0.5 $CODE_DIR/scripts/metrics_params/baseline.csv"
alias watch_cider_csv="$CODE_DIR/scripts/watch_csv.sh 0.5 $CODE_DIR/scripts/metrics_params/base_pt.csv"

# export these to other shells
export -f tasklist_pid
export -f watch_nvsmi
export -f watch_nvsmi_simple

echo "Done! "
cd $CODE_DIR