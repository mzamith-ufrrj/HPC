#readfile <- read.csv("L1.sequencial.csv")
library(ggplot2)
#library(dplyr)
#library(magrittr) 
dfl1 <-read.table("L1.sequencial.csv", header=FALSE, sep= ";")
#print(mean(dfl1$V2[11:20]))
mem <- c(1:51)
ave_time <- c(1:51)
sd_time <- c(1:51)
for (i in 1:51){
  mem[i] <- mean(dfl1$V2[((i-1)*10)+1:(i*10)])
  ave_time[i] <- mean(dfl1$V3[((i-1)*10)+1:(i*10)])
  sd_time[i]  <- sd(dfl1$V3[((i-1)*10)+1:(i*10)])
  #print(sprintf("(%d, %d) -> %d", ((i-1)*10)+1,(i*10), mean(dfl1$V2[((i-1)*10)+1:10]) ))
  #print(mean(dfl1$V2[((i-1)*10)+1:(i*10)]))
  #print(((i-1)*10)+1)
  #print((i*10))
  #print(dfl1$V2)
}

elapsedTime_L1 <- data.frame(
  memory_used = mem,
  et_ave_seq = ave_time,
  et_sd_seq = sd_time,
  stringsAsFactors = FALSE
)
ggplot() +
  geom_point(data=elapsedTime_L1, aes(x=memory_used, y=et_ave_seq), color='blue') +
#geom_point(data=dfl2, aes(x=V2, y=V3), color='red') +
  labs(x = "Bytes", y = "Time in seconds")

mz@157#dfl2 <-read.table("L2.sequencial.csv", header=FALSE, sep= ";")

#print(df, n=51 )
