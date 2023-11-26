import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv ('bello.csv')
# print(df)


#####################################################
# Creating a plot for naive algorithm
n = np.unique(df['n'])
naive = df[df['AlgorithmID'] == 'naive']


plt.figure()
plt.plot(naive['n'], naive['time'])
plt.xlabel('n')
plt.ylabel('time [ms]')
plt.title('Naive MMM Time Complexity')
plt.savefig('naiveplt.png')