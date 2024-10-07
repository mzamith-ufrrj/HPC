library(readr)
library(ggplot2)
library(reshape2)
rm(list = ls())
print("Aceleração e eficiência método de Jacobi - dois processadores i7 de 4a geração")



cpu_i7_N_T1 <- read.table("01-log-i7note-1.csv", header=TRUE, sep= ";")
cpu_i7_N_Tn <- read.table("log-i7note.csv", header=TRUE, sep= ";")

cpu_xeon_T1 <- read.table("01-log-xeon-1.csv", header=TRUE, sep= ";")
cpu_xeon_Tn <- read.table("log-xeon.csv", header=TRUE, sep= ";")

cpu_i7_N_T1_media <- mean(cpu_i7_N_T1$tempo_total)
cat("i7-4510U @ 2.00GHz", cpu_i7_N_T1_media)
cpu_i7_N_Tn_media <- aggregate(x = cpu_i7_N_Tn$tempo_total, by=list(cpu_i7_N_Tn$threads), FUN=mean)
print(cpu_i7_N_Tn_media)
cpu_i7_N_Tn_Aceleracao <- cpu_i7_N_T1_media / cpu_i7_N_Tn_media$x
cpu_i7_N_Tn_eficiencia <- cpu_i7_N_Tn_Aceleracao * 1/cpu_i7_N_Tn_media$Group.1

cpu_xeon_T1_media <- mean(cpu_xeon_T1$tempo_total)
cpu_xeon_Tn_media <- aggregate(cpu_xeon_Tn$tempo_total, by=list(cpu_xeon_Tn$threads), FUN=mean)
cpu_xeon_Tn_Aceleracao <- cpu_xeon_T1_media / cpu_xeon_Tn_media$x
cpu_xeon_Tn_eficiencia <- cpu_xeon_Tn_Aceleracao * 1 / cpu_xeon_Tn_media$Group.1
cat("Xeon E-2226G @ 3.40GHz", cpu_xeon_T1_media)
print(cpu_xeon_Tn_media)
# g <- ggplot() +   
#   scale_x_continuous(breaks = seq(2, 12, by=1), limits=c(2,12))  +
#   geom_point(aes(x=cpu_xeon_Tn$threads, y=cpu_xeon_Tn$tempo_total, color="i7-4510U @ 2.00GHz"), shape=25, size=3) +
#   xlab(label = 'Quantidade de threads') +
#   ylab(label = 'Eficiência') + theme(legend.text = element_text(size=13),
#                                           legend.title=element_blank())
# g <- ggplot() +   
#   scale_x_continuous(breaks = seq(2, 12, by=1), limits=c(2,12))  +
#   geom_line(aes(x=cpu_i7_N_Tn_media$Group.1, y=cpu_i7_N_Tn_eficiencia, color="i7-4510U @ 2.00GHz"), size=0.5)  +
#   geom_point(aes(x=cpu_i7_N_Tn_media$Group.1, y=cpu_i7_N_Tn_eficiencia, color="i7-4510U @ 2.00GHz"), shape=25, size=3) +
#   geom_line(aes(x=cpu_xeon_Tn_media$Group.1, y=cpu_xeon_Tn_eficiencia, color="Xeon E-2226G @ 3.40GHz"), size=0.5)  +
#   geom_point(aes(x=cpu_xeon_Tn_media$Group.1, y=cpu_xeon_Tn_eficiencia, color="Xeon E-2226G @ 3.40GHz"), shape=25, size=3) +
#   
#   xlab(label = 'Quantidade de threads') +
#   ylab(label = 'Eficiência') + theme(legend.text = element_text(size=13),
#                                         legend.title=element_blank())
 
g <- ggplot() +   scale_x_continuous(breaks = seq(2, 12, by=1), limits=c(2,12))  +

  #geom_point(aes(x=memoria, y=aceleracao8x), shape=1, fill="deeppink", color="deeppink", size=3) +
  #geom_line(aes(x=memoria, y=aceleracao8x), color="deeppink", size=0.5)+
  #geom_point(aes(x=memoria, y=aceleracao7x, color="7 threads"), shape=0, size=3) +
  #geom_line(aes(x=memoria, y=aceleracao7x, color="7 threads"), size=0.5)+
  #geom_point(aes(x=memoria, y=aceleracao6x, color="6 threads"), shape=21, size=3) +
  #geom_line(aes(x=memoria, y=aceleracao6x, color="6 threads"), size=0.5)+
  #geom_point(aes(x=memoria, y=aceleracao5x, color="5 threads"), shape=22, size=3) +
  #geom_line(aes(x=memoria, y=aceleracao5x, color="5 threads"), size=0.5)+
  #geom_point(aes(x=memoria, y=aceleracao4x, color="4 threads"), shape=23, size=3) +
  #geom_line(aes(x=memoria, y=aceleracao4x, color="4 threads"), size=0.5)+
  #geom_point(aes(x=memoria, y=aceleracao2x, color="2 threads"), shape=24,size=3) +
  #geom_line(aes(x=memoria, y=aceleracao2x, color="2 threads"), size=0.5)+
  #geom_point(aes(x=memoria, y=aceleracao3x, color="3 threads"), shape=25, size=3) +

geom_line(aes(x=cpu_i7_N_Tn_media$Group.1, y=cpu_i7_N_Tn_Aceleracao, color="i7-4510U @ 2.00GHz"), size=0.8)  +
geom_point(aes(x=cpu_i7_N_Tn_media$Group.1, y=cpu_i7_N_Tn_Aceleracao, color="i7-4510U @ 2.00GHz"), shape=17, size=5) +
geom_line(aes(x=cpu_xeon_Tn_media$Group.1, y=cpu_xeon_Tn_Aceleracao, color="Xeon E-2226G @ 3.40GHz"), size=0.80)  +
geom_point(aes(x=cpu_xeon_Tn_media$Group.1, y=cpu_xeon_Tn_Aceleracao, color="Xeon E-2226G @ 3.40GHz"), shape=20, size=5) +
scale_color_manual(values=c('blue', 'red')) +
xlab(label = 'Quantidade de threads') +
ylab(label = 'Aceleração') + theme(legend.text = element_text(size=15),
                                   axis.text=element_text(size=15),
                                   axis.title=element_text(size=15),
                                   legend.title=element_blank(), 
                                   panel.background = element_rect(fill = "white",
                                                                   colour = "black",
                                                                   size = 0.5, linetype = "solid"),
                                   panel.grid.major = element_line(size = 0.5, linetype = 'dashed',
                                                                   colour = "lightgray"), 
                                   panel.grid.minor = element_line(size = 0.25, linetype = 'dashed',
                                                                   colour = "lightgray")
                                   )
print(g)