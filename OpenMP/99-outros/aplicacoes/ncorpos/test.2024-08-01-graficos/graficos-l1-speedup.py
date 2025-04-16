import pandas as pd
import statistics
import matplotlib.pyplot as plt
import sys
if __name__ == '__main__':
    print("Gráficos testes n-corpos")
    df_cpu_seqL1 = pd.read_csv("L1.sequencial-100S.csv", sep = ";")
    df_cpu_mpL1 = pd.read_csv("L1.openMP-100S.csv", sep = ";")
    #print(df_cpu_seqL1)
    bytes_seq = []
    speed_up = []
    
    for i in range (0, len(df_cpu_seqL1)):
        df_seq = df_cpu_seqL1.iloc[i]
        df_mp  = df_cpu_mpL1.iloc[i]
        bytes_seq.append(float(df_seq["memoryUsed"])/1024)
        seq = float(df_seq["totalTime"])
        omp = float(df_mp["totalTime"])
        speed_up.append(seq/omp)

       

    fig, ax =  plt.subplots(figsize=(12, 8))
    ax.plot(bytes_seq, speed_up, 'bo',  c='black')
    plt.xlabel('KBytes')
    plt.ylabel('Speedup')
    plt.legend()
    plt.title("Simulação de n-corpos CPU 1 thread x 6 threads")
    plt.show()