#!/bin/bash
cd ${0%/*}
kill -9 `cat logger.pid`
rm -f logger.pid
rm -f nohup.out
