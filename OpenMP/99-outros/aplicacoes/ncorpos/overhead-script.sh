#!/bin/bash
#for ((part = 1024; part <= 134217728; part*=2))
for ((part = 32; part <= 1024; part*=2))
do

	for ((s = 1; s <= 10; s+=1))#
	do
		echo "$part  step: $s"
		./memory.exec 3145728 1024 > /dev/null
		./n-bodies-cuda.exec 100 $part GTX1660.csv 0 > /dev/null
		./n-bodies-1.0.exec 100 $part XEON2226G-1T.csv 0 > /dev/null
		./n-bodies-2.0.exec 100 $part XEON2226G-6T.csv 0 > /dev/null
	done
done

