#!/bin/bash

mkdir fastqs

for i in {1..10}
do
    echo "This is sample number $i" > fastqs/sample_${i}.fastq
done
