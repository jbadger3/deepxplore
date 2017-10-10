#!/bin/bash
#to view all listening ports
#pdsh netstat -tulpn | grep LISTEN
pdsh pkill -f python
