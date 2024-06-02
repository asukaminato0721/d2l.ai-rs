#!/bin/bash
set -eux
mkdir -vp data
pushd ./data
wget -c "http://d2l-data.s3-accelerate.amazonaws.com/timemachine.txt"
printf "download finished\n"
