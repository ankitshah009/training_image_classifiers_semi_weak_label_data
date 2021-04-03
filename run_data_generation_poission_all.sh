#!/bin/bash

# Order of args is bagsize, ntrain, ntest, trials, beta, reuse

#bash run_data_generation_poisson.sh 8 10000 2000 5 0.5 2 
#bash run_data_generation_poisson.sh 8 5000 1000 5 0.5 2 
#bash run_data_generation_poisson.sh 16 10000 2000 5 0.5 2 
#bash run_data_generation_poisson.sh 16 5000 1000 5 0.5 2 
#bash run_data_generation_poisson.sh 16 4000 1000 5 2.0 2 
#bash run_data_generation_poisson.sh 32 8000 1600 5 1.2 8
bash run_data_generation_poisson.sh 2 10000 2000 5 0.5 2
bash run_data_generation_poisson.sh 4 10000 2000 5 0.5 2
