#!/bin/bash
cut -d',' -f2- $1 > temp.txt
sed -i '1d' temp.txt
awk '{for (i=1;i<=NF;i++) sum[i]+=$i} END{for (i in sum) print i, sum[i]}' temp.txt
