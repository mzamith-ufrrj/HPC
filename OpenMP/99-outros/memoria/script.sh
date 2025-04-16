#!/bin/bash

echo "Memória"
delta=10
for ((part = 10; part <= 1000; part+=delta))
do

	for ((s = 1; s <= 10; s+=1))#
	do
		echo "$part kb step: $s"
		./memory.exec $part 1000 64 0 l1.l2.malloc.csv > /dev/null
		./memory.exec $part 1000 64 1 l1.l2.posix_memalign.csv > /dev/null
	done
done

delta=100
for ((part = 2000; part <= 20000; part+=delta))
do

	for ((s = 1; s <= 10; s+=1))#
	do
		echo "$part kb step: $s"
		./memory.exec $part 1000 64 0 l3.malloc.csv > /dev/null
		./memory.exec $part 1000 64 1 l3.posix_memalign.csv > /dev/null
	done
done
