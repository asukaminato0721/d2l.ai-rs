#!/bin/sh
rm -rvf data
mkdir -vp data
pushd ./data
wget -c "https://huggingface.co/spaces/chrisjay/mnist-adversarial/resolve/820ea84d0cd6c3c937fe80c3ed849a16f693876b/files/MNIST/raw/t10k-images-idx3-ubyte"
wget -c "https://huggingface.co/spaces/chrisjay/mnist-adversarial/resolve/820ea84d0cd6c3c937fe80c3ed849a16f693876b/files/MNIST/raw/t10k-labels-idx1-ubyte"
wget -c "https://huggingface.co/spaces/chrisjay/mnist-adversarial/resolve/820ea84d0cd6c3c937fe80c3ed849a16f693876b/files/MNIST/raw/train-images-idx3-ubyte"
wget -c "https://huggingface.co/spaces/chrisjay/mnist-adversarial/resolve/820ea84d0cd6c3c937fe80c3ed849a16f693876b/files/MNIST/raw/train-labels-idx1-ubyte"
printf "download finished\n"
