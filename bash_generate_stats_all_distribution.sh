#!/bin/bash
root_path=$1

for i in {0..4};
do 
echo "train.trial\="${i}".csv"
bash generate_stats.sh $1/train.trial\=${i}.csv
done


for i in {0..4};
do 
echo "test.trial\="${i}".csv"
bash generate_stats.sh $1/test.trial\=${i}.csv
done

