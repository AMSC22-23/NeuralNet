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

    print(naive)

    # for each dataframe we extract just two values: the misses when datatype is 1 and the misses when datatype is 0
    miss_naive_float = naive[naive['datatype'] == 0]['misses']
    miss_naive_double = naive[naive['datatype'] == 1]['misses']

    miss_loopI_float = loopI[loopI['datatype'] == 0]['misses']
    miss_loopI_double = loopI[loopI['datatype'] == 1]['misses']

    miss_tiling_float = tiling[tiling['datatype'] == 0]['misses']
    miss_tiling_double = tiling[tiling['datatype'] == 1]['misses']

    miss_multiT_float = multiT[multiT['datatype'] == 0]['misses']
    miss_multiT_double = multiT[multiT['datatype'] == 1]['misses']

    # plotting the histogram
    fig = plt.figure()
    plt.hist(["naive_float", "naive_double", "loopI_float", "loopI_double", "tiling_float", "tiling_double", "multiT_float", "multiT_double"], [miss_naive_float, miss_naive_double, miss_loopI_float, miss_loopI_double, miss_tiling_float, miss_tiling_double, miss_multiT_float, miss_multiT_double])
    plt.title("Misses for each algorithm")
    plt.xlabel("Algorithm")
    plt.ylabel("# of misses")
    plt.show()


plot_histogram(data)