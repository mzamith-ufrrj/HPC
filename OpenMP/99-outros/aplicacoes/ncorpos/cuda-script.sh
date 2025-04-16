#!/bin/bash
rm -f 2024-08-03-CUDA-100S.csv
echo "Teste com CUDA"
#for ((part = 1024; part <= 134217728; part*=2))
delta=100
for ((part = 100; part <= 10000; part+=delta))
do

	for ((s = 1; s <= 10; s+=1))#
	do
		echo "$part  step: $s"
		./memory.exec 3145728 1024 > /dev/null
		./n-bodies-cuda.exec 100 $part 2024-08-03-CUDA-100S.csv 0 > /dev/null
	done
done
