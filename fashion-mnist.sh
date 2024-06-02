#!/bin/bash
set -eux
rm -rvf data
mkdir -vp data
pushd ./data
wget -c "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz"
wget -c "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz"
wget -c "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz"
wget -c "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz"
printf "download finished\n"
for f in *; do
	7z x "$f"
done
printf "extract finished\n"
