import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# import matplotlib.pyplot as plt
columns = ['author', 'id', 'matrix_dimension', 'datatype', 'time [ms]', 'tile_dim', 'misses']
# Read in the data
df = pd.read_csv('filResult.csv', names=columns, header=None)

# Inspecting the data
# print(df.head())


def correct_csv(dataframe: pd.DataFrame):
    """
    Corrects the csv file by swapping the values of misses and tile_dim when misses is NaN
    :param dataframe:
    :return: the corrected dataframe
    """

    data = dataframe.copy(deep=False)
    for i in range(len(data)):
        if np.isnan(data['misses'][i]):
            data['misses'][i] = data['tile_dim'][i]
            data['tile_dim'][i] = np.nan

    # exchanging the columns names
    dataframe.rename(columns={'misses': 'tile_dim', 'tile_dim': 'misses'}, inplace=True)

    return dataframe


df = correct_csv(df)
print(df.head())

def best_time(dataframe: pd.DataFrame):
    """
    Returns the best time for each matrix dimension and datatype
    :param dataframe:
    :return: a dataframe with the best time for each matrix dimension and datatype
    """
    best_time_df = pd.DataFrame(columns=dataframe.columns)
    for dim in dataframe['matrix_dimension'].unique():
        for datatype in dataframe['datatype'].unique():
            best_time_df = best_time_df._append(dataframe[(dataframe['matrix_dimension'] == dim) & (dataframe['datatype'] == datatype)].sort_values(by='time [ms]').head(1))
    return best_time_df


# print(best_time(df))


# plotting histogram: effect of misses over different algorithms

def plot_histogram(dataframe: pd.DataFrame, title: str):
    """
    Plots a histogram of the misses for each algorithm: the algorthms are:
    101: naive
    102: loopI
    103: tiling
    104: multiT

    :param dataframe: the dataframe containing the data
    :param title: the title of the histogram
    :return: None
    """

    # Extracting the data
    naive = dataframe[dataframe['id'] == '101']['misses']
    print(naive)
    loopI = dataframe[dataframe['id'] == '102']['misses']
    tiling = dataframe[dataframe['id'] == '103']['misses']
    multiT = dataframe[dataframe['id'] == '104']['misses']

    # Now each df contains the misses for each algorithm both for float and double, so we can plot them

    # Plotting the histogram
    fig, axs = plt.subplots(2, 2)
    fig.suptitle(title)
    axs[0, 0].hist(naive)
    axs[0, 0].set_title('Naive')
    axs[0, 1].hist(loopI)
    axs[0, 1].set_title('LoopI')
    axs[1, 0].hist(tiling)
    axs[1, 0].set_title('Tiling')
    axs[1, 1].hist(multiT)
    axs[1, 1].set_title('MultiT')
    plt.show()

    # saving the figure
    fig.savefig(title + '.png')

    return None

# Plotting the histogram
plot_histogram(df, 'Misses Histogram')


def plot_time_complexity():
    """
    Plots the time complexity of the algorithms: time vs matrix dimension
    :return: None
    """
    #extracting the data
    naive = df[df['id'] == 101]
    loopI = df[df['id'] == 102]
    tiling = df[df['id'] == 103]
    multiT = df[df['id'] == 104]

    # Plotting the time complexity in the same figure
    fig = plt.figure()
    plt.plot(naive['matrix_dimension'], naive['time [ms]'], label='Naive')
    plt.plot(loopI['matrix_dimension'], loopI['time [ms]'], label='LoopI')
    plt.plot(tiling['matrix_dimension'], tiling['time [ms]'], label='Tiling')
    plt.plot(multiT['matrix_dimension'], multiT['time [ms]'], label='MultiT')
    plt.legend()
    plt.xlabel('Matrix Dimension')
    plt.ylabel('Time [ms]')
    plt.title('Time Complexity')
    plt.show()

    # saving the figure
    fig.savefig('Time Complexity.png')

    return None
