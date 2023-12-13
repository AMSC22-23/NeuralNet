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


def misses_bar_plot():
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


    data = collection.find({'matrix_dimension': '1024X1024', 'datatype': 'float', 'misses': {'$ne': -1}, 'id': {'$in': ['101', '102', '103', '104', '105', '106', '107']}})

    data = pd.DataFrame(data)

    # we reduce now the data DataFrame by considering just one data for each id
    data = data.drop_duplicates(subset='id', keep='first')

    # we now plot the data
    plt.bar(data['id'], data['misses'])
    plt.xlabel('Algorithm')
    plt.ylabel('Misses')
    plt.title('Misses for each algorithm on square 1024 matrices on float datatype')

    # saving the plot
    plt.savefig('plot/misses_float.png')

    plt.close()

    # we now query all data where the dimension is 1024, datatype is double and the number of misses is not NaN or -1
    # moreover the id must be one of 101, 102, 103, 104, 105, 106, 107
    # we want just one data for each id
    data = collection.find({'matrix_dimension': '1024X1024', 'datatype': 'double', 'misses': {'$ne': -1}, 'id': {'$in': ['101', '102', '103', '104', '105', '106', '107']}})
    data = pd.DataFrame(data)

    # we reduce now the data DataFrame by considering just one data for each id
    data = data.drop_duplicates(subset='id', keep='first')

    # we now plot the data
    plt.bar(data['id'], data['misses'])
    plt.xlabel('Algorithm')
    plt.ylabel('Misses')
    plt.title('Misses for each algorithm on square 1024 matrices on double datatype')

    # saving the plot
    plt.savefig('plot/misses_double.png')

    plt.close()

misses_bar_plot()

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
    plt.show()
    plt.close()
    # saving the plot






time_complexity_plot('124')