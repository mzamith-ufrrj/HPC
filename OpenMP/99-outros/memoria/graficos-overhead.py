import pandas as pd
import statistics
import matplotlib.pyplot as plt
import sys
if __name__ == '__main__':
    print("Gráficos testes n-corpos")
    df_cpu_seqL1_00100S = pd.read_csv("L1.openMP-10000S.csv", sep = ";")
    #df_cpu_seqL1_01000S = pd.read_csv("L1.sequencial-1000S.csv", sep = ";")
    #df_cpu_seqL1_10000S = pd.read_csv("L1.sequencial-10000S.csv", sep = ";")
    #print(df_cpu_seqL1)
    bytes = []
    proce_av = []
    overh_av = []
    i = 0
    while i < len(df_cpu_seqL1_00100S):
        av_bytes = []
        av_proce = []
    
        for j in range(i, i+10):
            df = df_cpu_seqL1_00100S.iloc[j]
            tot = float(df["totalTime"])
            pro = float(df["processorTime"])
            av_bytes.append(float(df["memoryUsed"]))
            av_proce.append(pro/tot)

        bytes.append(statistics.mean(av_bytes) / 1024)
        av_proc = statistics.mean(av_proce)
        proce_av.append(av_proc)
        overh_av.append(1.0 - av_proc)
        i = i + 10
    
       

    fig, ax =  plt.subplots(figsize=(12, 8))
    ax.plot(bytes, proce_av, 'bo',  c='black')
    plt.xlabel('KBytes')
    plt.ylabel('Speedup')
    plt.legend()
    plt.title("Simulação de n-corpos CPU 1 thread x 6 threads")
    plt.show()