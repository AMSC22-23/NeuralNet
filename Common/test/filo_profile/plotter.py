import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# removing all warnings
import warnings
warnings.filterwarnings("ignore")

columns = ['author', 'id', 'matrix_dimension', 'datatype', 'time [ms]', 'tile_dim', 'misses']

# Read in the data
df = pd.read_csv('filResult.csv', names=columns, header=None)

# Corrects the csv file by swapping the values of misses and tile_dim when misses is NaN
data = df.copy()


for i in range(len(data)):
    if pd.isnull(data['misses'][i]):
        data['misses'][i] = data['tile_dim'][i]
        data['tile_dim'][i] = np.nan



def plot_histogram(dataframe):
    """
    Plots a histogram of the misses for each algorithm: the algorthms are: 101: naive 102: loopI 103: tiling 104: multiT
    :param dataframe: the original dataframe
    :return: nothing
    """

    # Extracting the data
    naive = dataframe[dataframe['id'] == '101']
    loopI = dataframe[dataframe['id'] == '102']
    tiling = dataframe[dataframe['id'] == '103']
    multiT = dataframe[dataframe['id'] == '104']

    # for each dataframe we extract just two values: the misses when datatype is 1 and the misses when datatype is 0
    miss_naive_float = naive[naive['datatype'] == 'float']['misses'].iloc[0]
    miss_naive_double = naive[naive['datatype'] == 'double']['misses'].iloc[0]
    miss_loopI_float = loopI[loopI['datatype'] == 'float']['misses'].iloc[0]
    miss_loopI_double = loopI[loopI['datatype'] == 'double']['misses'].iloc[0]
    miss_tiling_float = tiling[tiling['datatype'] == 'float']['misses'].iloc[0]
    miss_tiling_double = tiling[tiling['datatype'] == 'double']['misses'].iloc[0]
    miss_multiT_float = multiT[multiT['datatype'] == 'float']['misses'].iloc[0]
    miss_multiT_double = multiT[multiT['datatype'] == 'double']['misses'].iloc[0]


    # plotting the histogram
    fig = plt.figure()

    plt.bar(["naive_float", "naive_double", "loopI_float", "loopI_double", "tiling_float", "tiling_double", "multiT_float", "multiT_double"], [miss_naive_float, miss_naive_double, miss_loopI_float, miss_loopI_double, miss_tiling_float, miss_tiling_double, miss_multiT_float, miss_multiT_double])
    plt.title("Misses for each algorithm")
    plt.xlabel("Algorithm")
    plt.ylabel("# of misses")

    plt.show()
    plt.savefig("misses_algorithms.png")


plot_histogram(data)

def plot_time_complexity(dataframe):
    """
    Plots the time complexity of the algorithms. The algorithms are: 101: naive 102: loopI 103: tiling 104: multiT
    :param dataframe: the original dataframe
    :return: nothing
    """

    # Extracting the data
    naive = dataframe[dataframe['id'] == '101']
    loopI = dataframe[dataframe['id'] == '102']
    tiling = dataframe[dataframe['id'] == '103']
    multiT = dataframe[dataframe['id'] == '104']

    # for each dataframe we extract just two values: the time when datatype is 1 and the time when datatype is 0

    time_naive_float = naive[naive['datatype'] == 'float']['time [ms]'].iloc[0]
    time_naive_double = naive[naive['datatype'] == 'double']['time [ms]'].iloc[0]
    time_loopI_float = loopI[loopI['datatype'] == 'float']['time [ms]'].iloc[0]
    time_loopI_double = loopI[loopI['datatype'] == 'double']['time [ms]'].iloc[0]
    time_tiling_float = tiling[tiling['datatype'] == 'float']['time [ms]'].iloc[0]
    time_tiling_double = tiling[tiling['datatype'] == 'double']['time [ms]'].iloc[0]
    time_multiT_float = multiT[multiT['datatype'] == 'float']['time [ms]'].iloc[0]
    time_multiT_double = multiT[multiT['datatype'] == 'double']['time [ms]'].iloc[0]

    # plotting the histogram
    fig = plt.figure()
    plt.bar(["naive_float", "naive_double", "loopI_float", "loopI_double", "tiling_float", "tiling_double", "multiT_float", "multiT_double"], [time_naive_float, time_naive_double, time_loopI_float, time_loopI_double, time_tiling_float, time_tiling_double, time_multiT_float, time_multiT_double])
    plt.title("Time complexity for each algorithm on a 1024X1024 matrix")
    plt.xlabel("Algorithm")
    plt.ylabel("Time [ms]")
    plt.show()
    plt.savefig("time_complexity_algorithms.png")

plot_time_complexity(data)