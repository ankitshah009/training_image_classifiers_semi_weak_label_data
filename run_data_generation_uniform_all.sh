#!/bin/bash

# Order of args is bagsize, ntrain, ntest, trials, beta, reuse

bash run_data_generation_uniform.sh 8 10000 2000 5 3 2
bash run_data_generation_uniform.sh 16 4000 1000 5 3 2


