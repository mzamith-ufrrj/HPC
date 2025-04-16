#!/bin/bash
#rm -f shared.csv
echo "Teste com CUDA"
#for ((part = 1024; part <= 134217728; part*=2))

for ((part = 32; part <= 65536; part*=2))
do

	for ((s = 1; s <= 10; s+=1))#
	do
		echo "$part  step: $s"
		./GOL-Benchmarking.exec 8192 $part  100 single-height-cpu_s.csv cpu_s 0 > /dev/null
		./GOL-Benchmarking.exec 8192 $part  100 multithread-height-cpu_m.csv cpu_m 0 > /dev/null
	done
done

