#!/bin/bash
CSV=${2:-./metrics_params/base_pt.csv}
watch -n `[ -x $1 ] && echo 0.5 || echo $1` "sed -n '{1p;\$p}' $CSV" 
