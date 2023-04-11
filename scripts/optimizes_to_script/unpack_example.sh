#!/bin/bash

for tuple in "1,2.0,3" "4,5.5,6" "7,8,9.9"
do
    IFS=',' read -r x y z <<< "${tuple}"
    new_x=${x}
    new_y=${y}
    new_z=${z}
    echo "${new_x} ${new_y} ${new_z}"
done
