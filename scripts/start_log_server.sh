#!/bin/bash
cd ${0%/*}
nohup python logger_server.py &> /dev/null &
