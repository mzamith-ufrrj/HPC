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
    bytes_red = []
    proce_ave = []
    overh_ave = []
    proce_std = []
    overh_std = []
    i = 0
    while i < len(df_cpu_seqL1_00100S):
        samp_bytes = []
        samp_proce = []
        samp_overh = []
        
        for j in range(i, i+10):
            df = df_cpu_seqL1_00100S.iloc[j]
            #tot = float(df["totalTime"])
            pro = float(df["processorTime"])
            ove = float(df["overheadTime"])
            samp_bytes.append(float(df["memoryUsed"]))
            samp_proce.append(float(df["processorTime"]))
            samp_overh.append(float(df["overheadTime"]))
            

        bytes_red.append(statistics.mean(samp_bytes) / 1024)
        proce_ave.append(statistics.mean(samp_proce))
        overh_ave.append(statistics.mean(samp_overh))
        proce_std.append(statistics.stdev(samp_proce))
        overh_std.append(statistics.stdev(samp_overh))
        
        i = i + 10
    
       

    fig, ax =  plt.subplots(figsize=(12, 8))
    #ax.plot(bytes_red, proce_ave, 'bo',  c='black')
    width = 0.35  

    p1 = plt.bar(bytes_red, proce_ave, width, yerr = proce_std)
    p2 = plt.bar(bytes_red, overh_ave, width, bottom = proce_ave, yerr = overh_std)
    plt.xlabel('KBytes')
    plt.ylabel('Time (s)')
    plt.legend()
    plt.title("Simulação de n-corpos CPU 1 thread x 6 threads")
    plt.show()