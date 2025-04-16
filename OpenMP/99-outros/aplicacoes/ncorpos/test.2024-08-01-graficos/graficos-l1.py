import pandas as pd
import statistics
import matplotlib.pyplot as plt
import sys
if __name__ == '__main__':
    print("Gráficos testes n-corpos")
    df_cpu_seqL1 = pd.read_csv("L1.sequencial.csv", sep = ";")
    df_cpu_mpL1 = pd.read_csv("L1.openMP.csv", sep = ";")
    #print(df_cpu_seqL1)
    bytes_seq = []
    elapsedTime_seq = []
    
    elapsedTime_mp = []
    i = 0
    while i < len(df_cpu_seqL1):
        ave_bytes_seq = []
        ave_et_seq = []

        
        ave_et_mp = []
        for j in range(i, i+10):
            df_seq = df_cpu_seqL1.iloc[j]
            df_mp  = df_cpu_mpL1.iloc[j]
            ave_bytes_seq.append(float(df_seq["bytes"]))
            ave_et_seq.append(float(df_seq["time"]))
            ave_et_mp.append(float(df_mp["time"]))

        bytes_seq.append(statistics.mean(ave_bytes_seq) / 1024)
        elapsedTime_seq.append(statistics.mean(ave_et_seq))
        elapsedTime_mp.append(statistics.mean(ave_et_mp))
        i = i + 10

    fig, ax =  plt.subplots(figsize=(12, 8))
    ax.plot(bytes_seq, elapsedTime_seq, 'bo', label='Sequencial',  c='black')
    ax.plot(bytes_seq, elapsedTime_mp, 'bo', label='OpenMP (6 threads)',  c='blue')
    plt.xlabel('KBytes')
    plt.ylabel('Tempo (s)')
    plt.legend()
    plt.title("Simulação de n-corpos CPU 1 thread x 6 threads")
    plt.show()