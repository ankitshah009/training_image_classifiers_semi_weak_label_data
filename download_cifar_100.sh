#!/bin/bash
#wget -c https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
#tar -zxvf cifar-100-python.tar.gz
mkdir -p data
mv cifar-100-python/ data/
rm -rf cifar-100-python.tar.gz
