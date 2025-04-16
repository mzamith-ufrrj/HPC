import pandas as pd
import statistics
import matplotlib.pyplot as plt
import sys
if __name__ == '__main__':
    print("Cache")
    #df_malloc   = pd.read_csv("l1.l2.malloc.csv", sep = ";")
    #df_memali   = pd.read_csv("l1.l2.posix_memalign.csv", sep = ";")

    df_malloc   = pd.read_csv("l3.malloc.csv", sep = ";")
    df_memali   = pd.read_csv("l3.posix_memalign.csv", sep = ";")

    #print(df_cpu_seqL1)


    fig, ax =  plt.subplots(figsize=(12, 8))
    #ax.plot((df_malloc["MEMORY_USED"] /1024), (df_malloc["PAPI_L1_TCM"] + df_malloc["PAPI_L2_TCM"] + df_malloc["PAPI_L3_TCM"]) / df_malloc["MEMORY_USED"], 'bo', label='malloc',  c='black')
    #ax.plot((df_memali["MEMORY_USED"] /1024), (df_memali["PAPI_L1_TCM"] + df_memali["PAPI_L2_TCM"] + df_memali["PAPI_L3_TCM"]) / df_malloc["MEMORY_USED"], 'bo', label='posix_memalign',  c='blue')
    ax.plot((df_malloc["MEMORY_USED"] /1024), (df_malloc["PAPI_L2_TCA"] + df_malloc["PAPI_L3_TCA"]) / df_malloc["MEMORY_USED"], 'bo', label='malloc',  c='black')
    ax.plot((df_memali["MEMORY_USED"] /1024), (df_memali["PAPI_L2_TCA"] + df_memali["PAPI_L3_TCA"]) / df_malloc["MEMORY_USED"], 'bo', label='posix_memalign',  c='blue')
    #ax.plot(bodies, elapsedTime_gpu, 'bo', label='GTX1660 CUDA',  c='green')
    plt.xscale('log',base=2)
    plt.yscale('log',base=10)
    plt.xlabel('KBytes')
    plt.ylabel('Falha de cache por byte')
    plt.legend()
    plt.title("Memória Intel(R) Xeon(R) E-2226G x GTX1660")
    plt.show()
