#!/bin/bash

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

# export these to other shells
export -f tasklist_pid
export -f watch_nvsmi
export -f watch_nvsmi_simple

