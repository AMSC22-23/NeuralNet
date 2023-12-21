import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pymongo import MongoClient
from dbconnection import upload_new_data

def plot_effect_of_different_tilesize(collection):
    """
    This function plot the effect of different tilesize on the time complexity of the tiling algorithm

    :return:
    """

    # extracting the data from the database
    data = collection.find({'id': '128', 'datatype': 'float', 'matrix_dimension': '1024X1024', 'misses': {'$ne': -1}})
    data = pd.DataFrame(data)



    # Converting the time column to float
    data['time [ms]'] = data['time [ms]'].astype(float)
    # Converting the tilesize column to int, using a lambda function
    data['tile_dim'] = data['tile_dim'].apply(lambda x: int(x))
    # Convert the misses column to int
    data['misses'] = data['misses'].astype(int)
    # Sorting the data by tilesize
    data = data.sort_values(by=['tile_dim'])

    # keeping just one data for each tilesize
    data = data.drop_duplicates(subset='tile_dim', keep='first')

    # dropping the entries where tilsize is greater than 128
    data = data[data['tile_dim'] <= 128]

    # Plotting the data

    # creating a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)


    # plotting the time complexity
    # we use also dots to better see the data
    ax1.plot(data['tile_dim'], data['time [ms]'], '-o', label='time complexity')

    ax1.set_xlabel('tile_dim')
    ax1.set_ylabel('time [ms]')
    ax1.set_title('Effect of different tile dimensions on 1024X1024 matrices (float)')


    # addding a label to each dot: the tilesize

    for i, txt in enumerate(data['tile_dim']):
        if i == 0 or i == 1:
            ax1.annotate(txt, (data['tile_dim'].iloc[i], data['time [ms]'].iloc[i]),
                         xytext=(6, -3), textcoords='offset points', ha='left',  rotation=0)
        else:
            ax1.annotate(txt, (data['tile_dim'].iloc[i], data['time [ms]'].iloc[i]),
                         xytext=(0, 5), textcoords='offset points', ha='center',  rotation=0)


    ax1.grid()
    ax1.legend()

    # plotting the number of misses
    ax2.plot(data['tile_dim'], data['misses'],'-o' ,label='misses')
    ax2.set_xlabel('tile_dim')
    ax2.set_ylabel('misses')
    #ax2.set_title('Time complexity of tiling algorithm with different tilesize')

    for i, txt in enumerate(data['tile_dim']):
        if i>2:
            ax2.annotate(txt, (data['tile_dim'].iloc[i], data['misses'].iloc[i]), xytext=(3, -10), textcoords='offset points', ha='center',  rotation=0)
        else:
            ax2.annotate(txt, (data['tile_dim'].iloc[i], data['misses'].iloc[i]), xytext=(6, -3), textcoords='offset points', ha='left',  rotation=0)


    ax2.legend()
    ax2.grid()

    #saving the plot
    plt.savefig('plot/NEWtilesize_effect_float.png')


    plt.close()

    # we now plot the same graph for the double datatype
    # extracting the data from the database
"""
    data = collection.find({'id': '103', 'datatype': 'double', 'matrix_dimension': '1024X1024', 'misses': {'$ne': -1}})
    data = pd.DataFrame(data)

    # Converting the time column to float
    data['time [ms]'] = data['time [ms]'].astype(float)
    # Converting the tilesize column to int, using a lambda function
    data['tile_dim'] = data['tile_dim'].apply(lambda x: int(x))
    # Convert the misses column to int
    data['misses'] = data['misses'].astype(int)
    # Sorting the data by tilesize
    data = data.sort_values(by=['tile_dim'])

    # keeping just one data for each tilesize
    data = data.drop_duplicates(subset='tile_dim', keep='first')

    # Plotting the data

    # creating a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)


    # plotting the time complexity
    # we use also dots to better see the data
    ax1.plot(data['tile_dim'], data['time [ms]'], '-o', label='time complexity')

    ax1.set_xlabel('tile_dim')
    ax1.set_ylabel('time [ms]')
    ax1.set_title('Effect of different tile dimensions on 1024X1024 matrices (double)')


    # addding a label to each dot: the tilesize

    for i, txt in enumerate(data['tile_dim']):
        if i == 0 or i == 1:
            ax1.annotate(txt, (data['tile_dim'].iloc[i], data['time [ms]'].iloc[i]),
                         xytext=(6, -3), textcoords='offset points', ha='left', rotation=0)
        else:
            ax1.annotate(txt, (data['tile_dim'].iloc[i], data['time [ms]'].iloc[i]),
                         xytext=(0, 5), textcoords='offset points', ha='center', rotation=0)


    ax1.grid()
    ax1.legend()

    # plotting the number of misses
    ax2.plot(data['tile_dim'], data['misses'], '-o', label='misses')
    ax2.set_xlabel('tile_dim')
    ax2.set_ylabel('misses')
    # ax2.set_title('Time complexity of tiling algorithm with different tilesize')

    for i, txt in enumerate(data['tile_dim']):
        if i > 2:
            ax2.annotate(txt, (data['tile_dim'].iloc[i], data['misses'].iloc[i]), xytext=(3, -10),
                         textcoords='offset points', ha='center', rotation=0)
        else:
            ax2.annotate(txt, (data['tile_dim'].iloc[i], data['misses'].iloc[i]), xytext=(6, -3),
                         textcoords='offset points', ha='left', rotation=0)


    ax2.legend()
    ax2.grid()

    # saving the plot
    plt.savefig(' plot/tilesize_effect_double.png')
    """

client = MongoClient('mongodb://localhost:27017/')
db = client['AMSC_PROJECT']  # Cambia con il nome del tuo database
collection = db['filresult']
upload_new_data()

plot_effect_of_different_tilesize(collection)
