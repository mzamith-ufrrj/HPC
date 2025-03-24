rm(list=ls())
print("Resultados da multiplicação de matrizes")
gpu <- read.table("resultados.csv", header = T, 
                  row.names = NULL, 
                  stringsAsFactors = FALSE, sep=";")
gpu$bytesTrafegados <- gpu$bytesTrafegados / 1024
gpu$bytesTrafegados <- gpu$bytesTrafegados / 1024
rotulo <- c("global", "shared")
g <- ggplot()+
  geom_point(aes(x=gpu$bytesTrafegados, y=gpu$global, color=rotulo[1], fill = rotulo[1]), shape=22, size=5) +
  geom_line(aes(x=gpu$bytesTrafegados, y=gpu$global, color=rotulo[1]), size=0.85) + 
  geom_point(aes(x=gpu$bytesTrafegados, y=gpu$shared, color=rotulo[2], fill = rotulo[2]), shape=22, size=5) +
  geom_line(aes(x=gpu$bytesTrafegados, y=gpu$shared, color=rotulo[2]), size=0.85) +
  scale_x_continuous(trans='log2') +
  scale_y_continuous(trans='log10') +
  xlab(label = "Megabytes trafegados") +
  ylab(label = "Tempo (segundos)") + theme(legend.text = element_text(size=20),
                                           axis.text= element_text(size=16),
                                           axis.title = element_text(size=18),
                                     legend.title=element_blank())
print(g)