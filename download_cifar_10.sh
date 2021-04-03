#!/bin/bash
wget -c https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar -zxvf cifar-10-python.tar.gz
mkdir -p data
mv cifar-10-batches-py/ data/
rm -rf cifar-10-python.tar.gz
