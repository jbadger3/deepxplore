#!/bin/bash

num_steps=$1

#start param server on vm-3-5 vm-3-1 will be chief
ssh vm-3-5 python ~/project/cs744_project_d3/distributed_test/distributed_test.py --job_name ps --task_index 0 --num_steps $num_steps &
for vm in 1 2 3 4;
do
  let task="$vm - 1"
  ssh "vm-3-$vm" python ~/project/cs744_project_d3/distributed_test/distributed_test.py --job_name worker --task_index $task --num_steps $num_steps &
done
# start cheif on vm-3-1
#python ~/project/cs744_project_d3/distributed_test/distributed_test.py --job_name worker --task_index 0 --num_steps $num_steps
