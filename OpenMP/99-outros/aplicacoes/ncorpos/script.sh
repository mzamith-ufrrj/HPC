#!/bin/bash
rm -f 2024-08-01-Sequencial-100S.csv
echo "Teste com OpenMP"
delta=100

for ((part = 100; part <= 10000; part+=delta))
do

	for ((s = 1; s <= 10; s+=1))#
	do
		echo "$part  step: $s"
		./memory.exec 3145728 1024 > /dev/null
		./n-bodies-1.0.exec 100 $part 2024-08-01-Sequencial-100S.csv 0 > /dev/null
	done
done
