# this script will create different plot from data stored in mongodb

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# removing all warnings
import warnings
warnings.filterwarnings("ignore")

#conncetion to the database
from pymongo import  MongoClient
from dbconnection import upload_new_data



# we first upload the new data from the csv file
upload_new_data()


# we now connect to the database
client = MongoClient('mongodb://localhost:27017/')
db = client['AMSC_PROJECT']  # Cambia con il nome del tuo database
collection = db['filresult']


def id_to_name(ids):
    """
    This function map the id of the algorithm to its name:
    id         name
    1X1 = 1 -> naive
    1X2 = 2 -> loopI
    1X3 = 3 -> tiling
    1X4 = 4 -> multiT
    1X5 = 5 -> openblas
    1x6 = 6 -> avx
    1x7 = 7 -> avxT

    if the id has last digit equal to 1 it means that is naive,
    if the id has last digit equal to 2 it means that is loopI,
    and so on

    :param id:
    :return:
    """
    # we convert ids to a list of string(initially is a pd dataframe)
    ids = list(ids)

    result = []
    for ID in ids:

        last_digit = int(ID[-1])
        if last_digit == 1:
            result.append('naive')
        elif last_digit == 2:
            result.append('loopI')
        elif last_digit == 3:
            result.append('tiling')
        elif last_digit == 4:
            result.append('multiT')
        elif last_digit == 5:
            result.append('openblas')
        elif last_digit == 6:
            result.append('avx')
        elif last_digit == 7:
            result.append('avxT')
        else:
            result.append('unknown')
    return result

def misses_bar_plot(include_naive=False):
    """
    This function plot a bar plot using the data passed as argument
    :param data: the data to plot
    :param x_label: the label of the x axis
    :param y_label: the label of the y axis
    :param title: the title of the plot
    :return: None
    """

    # we'll now build a bar plot that shows the misses for each algorithm when the dimension is 1024.
    # we'll make one plot for each datatype (so 2 plot in total)
    # we'll use the data from the database

    # we now query all data where the dimension is 1024, datatype is float and the number of misses is not NaN or -1
    # moreover the id must be one of 101, 102, 103, 104, 105, 106, 107
    # we want just one data for each id

    if not include_naive:
        data = collection.find({'matrix_dimension': '1024X1024', 'datatype': 'float', 'misses': {'$ne': -1}, 'id': {'$in': ['101', '102', '103', '104', '105', '106', '107']}})
    else:
        data = collection.find({'matrix_dimension': '1024X1024', 'datatype': 'float', 'misses': {'$ne': -1}, 'id': {'$in': ['102', '103', '104', '105', '106', '107', '108']}})

    data = pd.DataFrame(data)

    # we reduce now the data DataFrame by considering just one data for each id
    data = data.drop_duplicates(subset='id', keep='first')
    #print(data['id'])
    # we now plot the data
    plt.bar(id_to_name(data['id']), data['misses'])
    plt.xlabel('Algorithm')
    plt.ylabel('Misses')
    plt.title('Misses for each algorithm on square 1024 matrices on float datatype')

    # saving the plot
    if not include_naive:
        plt.savefig('plot/misses_float.png')
    else:
        plt.savefig('plot/misses_float_no_naive.png')


    plt.close()

    # we now query all data where the dimension is 1024, datatype is double and the number of misses is not NaN or -1
    # moreover the id must be one of 101, 102, 103, 104, 105, 106, 107
    # we want just one data for each id
    data = collection.find({'matrix_dimension': '1024X1024', 'datatype': 'double', 'misses': {'$ne': -1}, 'id': {'$in': ['101', '102', '103', '104', '105', '106', '107']}})
    data = pd.DataFrame(data)

    # we reduce now the data DataFrame by considering just one data for each id
    data = data.drop_duplicates(subset='id', keep='first')

    # we now plot the data
    plt.bar(id_to_name(data['id']), data['misses'])
    plt.xlabel('Algorithm')
    plt.ylabel('Misses')
    plt.title('Misses for each algorithm on square 1024 matrices on double datatype')

    # saving the plot
    plt.savefig('plot/misses_double.png')


    plt.close()
    """
    # we now plot in a single figure the misses for each algorithm on square 1024 matrices on float and double datatype
    # we now query all data where the dimension is 1024, datatype is float and the number of misses is not NaN or -1
    # moreover the id must be one of 101, 102, 103, 104, 105, 106, 107
    # we want just one data for each id
    data = collection.find({'matrix_dimension': '1024X1024', 'datatype': 'float', 'misses': {'$ne': -1}, 'id': {'$in': ['101', '102', '103', '104', '105', '106', '107']}})
    data = pd.DataFrame(data)

    # we reduce now the data DataFrame by considering just one data for each id
    data = data.drop_duplicates(subset='id', keep='first')

    # we now plot the data
    plt.bar(id_to_name(id_to_name(data['id'])), data['misses'], label='float datatype')
    plt.xlabel('Algorithm')
    plt.ylabel('Misses')
    plt.title('Misses for each algorithm on square 1024 matrices on float datatype')

    
"""

#misses_bar_plot()

#misses_bar_plot(include_naive=True)

def time_complexity_plot(algorithm_id):
    """
    This function plot the time complexity of the algorithm passed as argument
    :param algorithm_id: the id of the algorithm
    :return: None
    """

    # we now query all data where the id is equal to the one passed as argument
    # we want just one data for each matrix dimension
    # we initally queru just the float datatype then we will do the same for the double datatype

    data = collection.find({'id': algorithm_id, 'datatype': 'float'})
    data = pd.DataFrame(data)
    data = data.drop_duplicates(subset='matrix_dimension', keep='first')


    # Converting the time column to float
    data['time [ms]'] = data['time [ms]'].astype(float)
    # Converting the matrix_dimension column to int, using a lambda function
    data['matrix_dimension'] = data['matrix_dimension'].apply(lambda x: int(x.split('X')[0]))
    # Sorting the data by matrix_dimension
    data = data.sort_values(by=['matrix_dimension'])
    # Plotting the data
    plt.plot(data['matrix_dimension'], data['time [ms]'], label='float datatype')
    plt.xlabel('Matrix dimension')
    plt.ylabel('time [ms]')
    plt.title('Time complexity of algorithm ' + algorithm_id )

    # doing the same thing for the double datatype
    data = collection.find({'id': algorithm_id, 'datatype': 'double'})
    data = pd.DataFrame(data)
    data = data.drop_duplicates(subset='matrix_dimension', keep='first')

    # Converting the time column to float
    data['time [ms]'] = data['time [ms]'].astype(float)
    # Converting the matrix_dimension column to int, using a lambda function
    data['matrix_dimension'] = data['matrix_dimension'].apply(lambda x: int(x.split('X')[0]))
    # Sorting the data by matrix_dimension
    data = data.sort_values(by=['matrix_dimension'])
    # Plotting the data
    plt.plot(data['matrix_dimension'], data['time [ms]'], label='double datatype')
    plt.xlabel('Matrix dimension')
    plt.ylabel('time [ms]')
    plt.title('Time complexity of algorithm ' + algorithm_id )


    # adding a grid to the plot
    plt.grid()

    # saving the plot
    plt.legend()
    plt.savefig('plot/time_complexity_' + algorithm_id + '.png')

    plt.close()
    # saving the plot

#time_complexity_plot('124')

def plot_effect_of_different_tilesize():
    """
    This function plot the effect of different tilesize on the time complexity of the tiling algorithm

    :return:
    """

    # extracting the data from the database
    data = collection.find({'id': '103', 'datatype': 'float', 'matrix_dimension': '1024X1024', 'misses': {'$ne': -1}})
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
    plt.savefig('plot/tilesize_effect_float.png')


    plt.close()

    # we now plot the same graph for the double datatype
    # extracting the data from the database

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
    plt.savefig('plot/tilesize_effect_double.png')



#plt.close()



#plot_effect_of_different_tilesize()

def plot_time_complexity(dtype):

    """
    Plots the time complexity of the algorithm of specified datatype
    :param dtype:
    :return:
    """

    plt.close()
    plt.figure()
    # plotting the time complexity of the naive algorithm

    # extracting the data from the database
    for id in ['121', '122', '125']:
        data = collection.find({'id': id, 'datatype': dtype})
        data = pd.DataFrame(data)


        data['time [ms]'] = data['time [ms]'].astype(float)
        data['matrix_dimension'] = data['matrix_dimension'].apply(lambda x: int(x.split('X')[0]))
        # removing from data entries with matrix dimension equal greater then 2048
        data = data[data['matrix_dimension'] <= 2048]
        data = data.sort_values(by=['matrix_dimension'])
        data = data.drop_duplicates(subset='matrix_dimension', keep='first')


        plt.plot(data['matrix_dimension'], data['time [ms]'], label=id_to_name([id]))



    for id in ['123', '124']:
        data = collection.find({'id': id, 'datatype': dtype, 'tile_dim': 32, })
        data = pd.DataFrame(data)


        data['time [ms]'] = data['time [ms]'].astype(float)
        data['matrix_dimension'] = data['matrix_dimension'].apply(lambda x: int(x.split('X')[0]))
        # removing from data entries with matrix dimension equal greater then 2048
        data = data[data['matrix_dimension'] <= 2048]
        data = data.sort_values(by=['matrix_dimension'])
        data = data.drop_duplicates(subset='matrix_dimension', keep='first')


        plt.plot(data['matrix_dimension'], data['time [ms]'], label=id_to_name([id]))

    for id in ['126', '127']:
        data = collection.find({'id': id, 'datatype': dtype})
        data = pd.DataFrame(data)


        data['time [ms]'] = data['time [ms]'].astype(float)
        data['matrix_dimension'] = data['matrix_dimension'].apply(lambda x: int(x.split('X')[0]))
        # removing from data entries with matrix dimension equal greater then 2048
        data = data[data['matrix_dimension'] <= 1024]
        data = data.sort_values(by=['matrix_dimension'])
        data = data.drop_duplicates(subset='matrix_dimension', keep='first')


        plt.plot(data['matrix_dimension'], data['time [ms]'], label=id_to_name([id]))




    plt.grid()
    plt.legend()

    plt.xlabel('Matrix dimension')
    plt.ylabel('time [ms]')

    plt.title('Time complexity of algorithms on ' + dtype + ' datatype')

    plt.savefig('plot/time_complexity_' + dtype + '.png')

#plot_time_complexity('double')
#plot_time_complexity('float')


def plot_time_tiling_multiT(data_type):
    """
    This function plot the time complexity of the tiling and multiT algorithm
    :return:
    """

    ids = ['123', '124', '125']
    plt.figure()
    # extracting the data from the database
    for id in ids:
        if id != '125':
            data = collection.find({'id': id, 'datatype': data_type, 'tile_dim': 32})
        else:
            data = collection.find({'id': id, 'datatype': data_type})
        data = pd.DataFrame(data)
        # for each matrix dimension we keep the data where the time is the minimum
        data = data.loc[data.groupby('matrix_dimension')['time [ms]'].idxmin()]
        # convert the time column to float
        data['time [ms]'] = data['time [ms]'].astype(float)
        # convert the matrix dimension column to int
        data['matrix_dimension'] = data['matrix_dimension'].apply(lambda x: int(x.split('X')[0]))
        # we sort the data by matrix dimension
        data = data.sort_values(by=['matrix_dimension'])
        # we now plot the data
        plt.plot(data['matrix_dimension'], data['time [ms]'], label=id_to_name([id]))



    plt.grid()
    plt.legend()
    plt.xlabel('Matrix dimension')
    plt.ylabel('time [ms]')
    plt.title('Time complexity of tiling and multiT algorithms on ' + data_type + ' datatype')
    plt.savefig('plot/time_complexity_tiling_multiT_' + data_type + '.png')
    plt.close()

plot_time_tiling_multiT('float')
plot_time_tiling_multiT('double')