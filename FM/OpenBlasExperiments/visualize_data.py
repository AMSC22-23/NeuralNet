import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv ('bello.csv')
# print(df)


#####################################################
# Creating a plot for naive algorithm 
n = np.unique(df['n'])
naive = df[df['AlgorithmID'] == 'naive']

blas = df[df['AlgorithmID'] == 'blas']


plt.figure()
plt.plot(naive['n'], naive['time'], label = 'naive')
plt.xlabel('n')
plt.ylabel('time [ms]')
plt.title('Naive MMM Time Complexity, precision = double')

plt.plot(blas['n'], blas['time'], label = 'blas')
plt.legend()
plt.savefig('naiveplt.png')