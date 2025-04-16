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
    elapsedTime_seq = []
    elapsedTime_mp = []
    for i in range (0, len(df_cpu_seqL1)):
        df_seq = df_cpu_seqL1.iloc[i]
        df_mp  = df_cpu_mpL1.iloc[i]
        bytes_seq.append(float(df_seq["memoryUsed"])/1024)
        elapsedTime_seq.append(float(df_seq["totalTime"]))
        elapsedTime_mp.append(float(df_mp["totalTime"]))

       

    fig, ax =  plt.subplots(figsize=(12, 8))
    ax.plot(bytes_seq, elapsedTime_seq, 'bo', label='Sequencial',  c='black')
    ax.plot(bytes_seq, elapsedTime_mp, 'bo', label='OpenMP (6 threads)',  c='blue')
    plt.xlabel('KBytes')
    plt.ylabel('Tempo (s)')
    plt.legend()
    plt.title("Simulação de n-corpos CPU 1 thread x 6 threads")
    plt.show()