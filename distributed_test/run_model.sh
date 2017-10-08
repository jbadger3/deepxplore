#!/bin/bash

script=$1
num_steps=$2

#start param server on vm-3-2 vm-3-1 will be chief
ssh vm-3-2 "'python $script --job_name ps --task_index 0 --num_steps $num_steps'"
for vm in 2 3 4 5;
do
  let task="$vm - 1"
  ssh vm-3-$vm "'python $script --job_name worker --task_index $task --num_steps $num_steps'"
done
python $script --job_name worker --task_index 0 --num_steps $num_steps
# start cheif on vm-3-1
