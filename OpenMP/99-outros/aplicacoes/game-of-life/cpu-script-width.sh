#!/bin/bash
#rm -f shared.csv
echo "Teste com CUDA"
#for ((part = 1024; part <= 134217728; part*=2))

for ((part = 32; part <= 65536; part*=2))
do

	for ((s = 1; s <= 10; s+=1))#
	do
		echo "$part  step: $s"
        ./GOL-Benchmarking.exec  $part 8192 100 single-width-cpu_s.csv cpu_s 0 > /dev/null
		./GOL-Benchmarking.exec  $part 8192 100 multithread-width-cpu_m.csv cpu_m 0 > /dev/null

	done
done

