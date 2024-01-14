import pandas as pd
import matplotlib.pyplot as plt
import struct

# Leggi il file CSV
df = pd.read_csv('Time_profile.csv')

#df["custom_x"] = (5 * df["n_layer1"] + 2 * ((df["n_layer1"] + 1) * df["n_layer1"]) + (df["n_layer1"] + 1) * 3)*struct.calcsize('f')/1024
df["custom_x"] = df["n_layer1"]
# Filtra i dati in base alle condizioni richieste

df_model_0 = df[df['model'] == 0]

df_model_3 = df[df['model'] == 3]

df_model_1_blk_size_4 = df[(df['model'] == 1) & (df['blk_size'] == 4)]
df_model_2_blk_size_4 = df[(df['model'] == 2) & (df['blk_size'] == 4)]

df_model_1_blk_size_8 = df[(df['model'] == 1) & (df['blk_size'] == 8)]
df_model_2_blk_size_8 = df[(df['model'] == 2) & (df['blk_size'] == 8)]

df_model_1_blk_size_16 = df[(df['model'] == 1) & (df['blk_size'] == 16)]
df_model_2_blk_size_16 = df[(df['model'] == 2) & (df['blk_size'] == 16)]

df_model_1_blk_size_32 = df[(df['model'] == 1) & (df['blk_size'] == 32)]
df_model_2_blk_size_32 = df[(df['model'] == 2) & (df['blk_size'] == 32)]

df_mode0_sorted = df_model_0.sort_values(by="n_layer1")

df_mode3_sorted = df_model_3.sort_values(by="n_layer1")

df_model_1_blk_size_4_sorted = df_model_1_blk_size_4.sort_values(by="n_layer1")
df_model_2_blk_size_4_sorted = df_model_2_blk_size_4.sort_values(by="n_layer1")

df_model_1_blk_size_16_sorted = df_model_1_blk_size_16.sort_values(by="n_layer1")
df_model_2_blk_size_16_sorted = df_model_2_blk_size_16.sort_values(by="n_layer1")

df_model_1_blk_size_8_sorted = df_model_1_blk_size_8.sort_values(by="n_layer1")
df_model_2_blk_size_8_sorted = df_model_2_blk_size_8.sort_values(by="n_layer1")

df_model_1_blk_size_32_sorted = df_model_1_blk_size_32.sort_values(by="n_layer1")
df_model_2_blk_size_32_sorted = df_model_2_blk_size_32.sort_values(by="n_layer1")

string = ["time_m1", "time_m2","time_m3", "time_m4","time_m5", "time_m6","time_m7", "time_m8","time_m9", "time_m10","time_m11", "time_predict", "time_batch", "time_rtn", "time_backprgtn"]
for value in string:

  time = value

  # Crea il grafico
  plt.figure(figsize=(13, 4))

  # Plot per model=0
  plt.plot(df_mode0_sorted['custom_x'], df_mode0_sorted[time]/1000, label='Sequential', marker='o')
  
  plt.plot(df_mode3_sorted['custom_x'], df_mode3_sorted[time]/1000, label='Optimized', marker='o')

  # Plot per model=1, blk_size=4
  plt.plot(df_model_1_blk_size_4_sorted['custom_x'], df_model_1_blk_size_4_sorted[time]/1000, label='CUDA 1, blk_size=4', marker='o')

  # Plot per model=2, blk_size=4
  plt.plot(df_model_2_blk_size_4_sorted['custom_x'], df_model_2_blk_size_4_sorted[time]/1000, label='CUDA 2, blk_size=4', marker='o')

  #plt.plot(df_model_1_blk_size_8_sorted['custom_x'], df_model_1_blk_size_8_sorted[time]/1000, label='Model 1, blk_size=8', marker='o')
  #plt.plot(df_model_1_blk_size_16_sorted['custom_x'], df_model_1_blk_size_16_sorted[time]/1000, label='Model 1, blk_size=16', marker='o')
  #plt.plot(df_model_1_blk_size_32_sorted['custom_x'], df_model_1_blk_size_32_sorted[time]/1000, label='Model 1, blk_size=32', marker='o')

  #plt.plot(df_model_2_blk_size_8_sorted['custom_x'], df_model_2_blk_size_8_sorted[time]/1000, label='Model 2, blk_size=8', marker='o')
  #plt.plot(df_model_2_blk_size_16_sorted['custom_x'], df_model_2_blk_size_16_sorted[time]/1000, label='Model 2, blk_size=16', marker='o')
  #plt.plot(df_model_2_blk_size_32_sorted['custom_x'], df_model_2_blk_size_32_sorted[time]/1000, label='Model 2, blk_size=32', marker='o')

  # Aggiungi etichette e titolo
  plt.xlabel('total parameters')
  l = ' ms'
  plt.ylabel(time+l)
  plt.title('Number of neurons per layer and time evaluated on different model')

  # Aggiungi la legenda
  plt.legend()
  #plt.xscale('log')

  # Visualizza il grafico
  plt.show()