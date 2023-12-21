import matplotlib.pyplot as plt
import pandas as pd
from pymongo import MongoClient
from dbconnection import upload_new_data
from plot_fromDB import id_to_name


def plot_time_complexity(dtype, collection):

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



    for id in ['123']:
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

    for id in ['128']:
        matrix_dims = ['256X256', '512X512', '1024X1024', '2048X2048']
        data = collection.find({'id': id, 'datatype': dtype, 'matrix_dimension': {'$in': matrix_dims}, 'threads': 64})
        data = pd.DataFrame(data)

        print(data)
        data['time [ms]'] = data['time [ms]'].astype(float)
        data['matrix_dimension'] = data['matrix_dimension'].apply(lambda x: int(x.split('X')[0]))
        # removing from data entries with matrix dimension equal greater then 2048
        data = data[data['matrix_dimension'] <= 2048]
        data = data.sort_values(by=['matrix_dimension'])
        data = data.drop_duplicates(subset='matrix_dimension', keep='first')

        plt.plot(data['matrix_dimension'], data['time [ms]'], label=id_to_name([id]))




    plt.grid()
    plt.legend()

    plt.xlabel('Matrix dimension')
    plt.ylabel('time [ms]')

    plt.title('Time complexity of algorithms on ' + dtype + ' datatype')

    plt.savefig('plot/NEWtime_complexity_' + dtype + '.png')



client = MongoClient('mongodb://localhost:27017/')
db = client['AMSC_PROJECT']  # Cambia con il nome del tuo database
collection = db['filresult']
upload_new_data()

plot_time_complexity('float', collection)
plot_time_complexity('double', collection)