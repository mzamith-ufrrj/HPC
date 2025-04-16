#https://r4ds.had.co.nz/data-import.html
library(readr)
library(ggplot2)
rm(list = ls())
print("Comparativo entre AMD e Intel(R) Xeon(R) E-2226G CPU @ 3.40GHz ")
#amd_s <- read_csv2("/home/mzamith/Documents/Develop/C++/HPC/exemplos/aplicacoes/ncorpos/test.2024-09-13-AMD/amd.sequencial.csv")
#amd_p <- read_csv2("/home/mzamith/Documents/Develop/C++/HPC/exemplos/aplicacoes/ncorpos/test.2024-09-13-AMD/amd.parallel.csv")
#xeon_s <- read_csv2("/home/mzamith/Documents/Develop/C++/HPC/exemplos/aplicacoes/ncorpos/test.2024-08-01-testes/L1.sequencial-100S.csv")
#xeon_p <- read_csv2("/home/mzamith/Documents/Develop/C++/HPC/exemplos/aplicacoes/ncorpos/test.2024-08-01-testes/L1.openMP-100S.csv")

amd_s <- read.table("/home/mzamith/Documents/Develop/C++/HPC/exemplos/aplicacoes/ncorpos/test.2024-09-13-AMD/amd.sequencial.csv", header=TRUE, sep= ";" )
amd_p <- read.table("/home/mzamith/Documents/Develop/C++/HPC/exemplos/aplicacoes/ncorpos/test.2024-09-13-AMD/amd.parallel.csv", header=TRUE, sep= ";" )
xeon_s <- read.table("/home/mzamith/Documents/Develop/C++/HPC/exemplos/aplicacoes/ncorpos/test.2024-08-01-testes/L1.sequencial-100S.csv", header=TRUE, sep= ";" )
xeon_p <- read.table("/home/mzamith/Documents/Develop/C++/HPC/exemplos/aplicacoes/ncorpos/test.2024-08-01-testes/L1.openMP-100S.csv", header=TRUE, sep= ";" )
amd_s$memoryUsed = amd_s$memoryUsed / 1024
xeon_s$memoryUsed = xeon_s$memoryUsed / 1024
amd_p$memoryUsed = amd_p$memoryUsed / 1024
xeon_p$memoryUsed = xeon_p$memoryUsed / 1024

g <- ggplot() +
  geom_point(data=amd_s, aes(x=amd_s$memoryUsed, y=amd_s$totalTime), color='blue') +
  geom_point(data=xeon_s, aes(x=xeon_s$memoryUsed, y=xeon_s$totalTime), color='red') +
  geom_point(data=amd_p, aes(x=amd_p$memoryUsed, y=amd_p$totalTime), color='green') +
  geom_point(data=xeon_p, aes(x=xeon_p$memoryUsed, y=xeon_p$totalTime), color='purple') +
  #scale_x_continuous(limits = c(20000, 40000)) + 
  #scale_y_discrete(breaks = seq(from = 0, to = 1, by = 0.1)) 
  
  #coord_trans(y ='log10', x='log2') +
  #scale_x_continuous(breaks = round(seq(min(xeon_s$memoryUsed), max(xeon_s$memoryUsed), by = 0.5),1)) +
  #scale_y_continuous(breaks = round(seq(min(xeon_s$totalTime), max(xeon_s$totalTime), by = 0.5),1))
  labs(x = "Bytes", y = "Time in seconds")

print(g)

