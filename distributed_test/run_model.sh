#!/bin/bash

model_script=$1
num_steps=$1

#start param server on vm-3-5
ssh vm-3-5 python ~/project/cs744_project_d3/distributed_test/distributed_test.py --job_name ps --task_index 0 --num_steps $num_steps &
# start cheif on vm-3-1
python ~/project/cs744_project_d3/distributed_test/distributed_test.py --job_name worker --task_index 0 --num_steps $num_steps &
#start workers on vm-3-2 - vm-3-4
ssh vm-3-2 python ~/project/cs744_project_d3/distributed_test/distributed_test.py --job_name worker --task_index 1 --num_steps $num_steps &
ssh vm-3-3 python ~/project/cs744_project_d3/distributed_test/distributed_test.py --job_name worker --task_index 2 --num_steps $num_steps &
ssh vm-3-4 python ~/project/cs744_project_d3/distributed_test/distributed_test.py --job_name worker --task_index 3 --num_steps $num_steps &

#loop not working.  Why?
#for vm in 2 3 4;
#do
#  task=$(($vm-1))
#  ssh vm-3-$vm python ~/project/cs744_project_d3/distributed_test/distributed_test.py --job_name worker --task_intex $task --num_steps $num_steps &
#done
