#!/bin/bash

# Argument order is Bagsize, nsample, batch size, epoch count, reuse, cuda, beta, binary_loss_weight, entropy_loss_weight, count_loss_weight, regularize, loss_type, classifier_type

#bash run_semi_supervised.sh 16 4000 128 100 2 0,1,2,3 2.0 1.0 0 1.0 0.01 "poisson" 0 0.01 Resnet18
bash run_semi_supervised.sh 32 4000 128 100 2 0,1,2,3 2.0 1.0 0 1.0 0.01 "poisson" 0 0.01 Resnet18
