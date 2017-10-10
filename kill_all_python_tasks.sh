#!/bin/bash
#to view all listening ports
#pdsh netstat -tulpn | grep LISTEN
for vm in 2 3 4 5
do
  echo "killing python processes in vm-3-$vm"
  ssh "vm-3-$vm" "pkill -f python && exit" || true
done
echo "killing python processes in vm-3-1"
pkill -f python
#pdsh pkill -f python
