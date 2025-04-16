import pandas as pd
import statistics
import matplotlib.pyplot as plt
import sys
if __name__ == '__main__':
    print("Gráficos testes n-corpos")
    df_cpu_seq  = pd.read_csv("2024-08-01-Sequencial-100S.csv", sep = ";")
    df_cpu_mp   = pd.read_csv("2024-08-01-OpenMP-100S.csv", sep = ";")
    df_cpu_cuda = pd.read_csv("2024-08-03-CUDA-100S.csv", sep = ";")
    #print(df_cpu_seqL1)
    bodies = []
    elapsedTime_seq = []
    elapsedTime_mp = []
    elapsedTime_gpu = []
    for i in range (0, len(df_cpu_seq)):
        df_seq    = df_cpu_seq.iloc[i]
        df_mp     = df_cpu_mp.iloc[i]
        df_cuda   = df_cpu_cuda.iloc[i]
        bodies.append(float(df_seq["memoryUsed"])/40)
        elapsedTime_seq.append(float(df_seq["totalTime"]))
        elapsedTime_mp.append(float(df_mp["totalTime"]))
        elapsedTime_gpu.append(float(df_cuda["totalTime"]))



    fig, ax =  plt.subplots(figsize=(12, 8))
    ax.plot(bodies, elapsedTime_seq, 'bo', label='Sequencial',  c='black')
    ax.plot(bodies, elapsedTime_mp, 'bo', label='OpenMP (6 threads)',  c='blue')
    ax.plot(bodies, elapsedTime_gpu, 'bo', label='GTX1660 CUDA',  c='green')
    plt.xlabel('Corpos')
    plt.ylabel('Tempo (s)')
    plt.legend()
    plt.title("Simulação de n-corpos Intel(R) Xeon(R) E-2226G x GTX1660")
    plt.show()
