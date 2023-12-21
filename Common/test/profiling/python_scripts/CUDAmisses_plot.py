import pandas as pd
import matplotlib.pyplot as plt
from pymongo import MongoClient

from dbconnection import upload_new_data


def CUDAmisses_plot(collection):

    #reading the csv file with header cuda/cudaprof.csv

    df = pd.read_csv('cuda/cudaprof.csv')

    # selecting only rows where block_size is 32
    df = df[df['block_size'] == 32]

    # query the database for openblas_data
    matrix_dims = ["1000X1000", "2000X2000", "3000X3000", "4000X4000", "5000X5000"]

    query = {"id": "125", "misses": -1, "matrix_dimension": {"$in": matrix_dims}}

    openblas_data = collection.find(query)
    openblas_data = pd.DataFrame(list(openblas_data))


    # we consider duplicates all the entries with the same matrix dimension
    openblas_data = openblas_data.sort_values(by=['matrix_dimension'])
    openblas_data = openblas_data.drop_duplicates(subset=['matrix_dimension'])
    # converting matrix_dimension to int using a lambda function
    openblas_data['matrix_dimension'] = openblas_data['matrix_dimension'].apply(lambda x: int(x.split('X')[0]))
    # sorting by matrix_dimension
    openblas_data = openblas_data.sort_values(by=['matrix_dimension'])
    # plotting

    plt.plot(openblas_data['matrix_dimension'], openblas_data['time [ms]'], label='openblas')


    plt.plot(df['matrix_dimension'], df['32_tile_time'], label='CUDA')

    plt.legend()
    plt.xlabel('matrix dimension')
    plt.ylabel('time [ms]')
    plt.grid()
    plt.title('Execution time vs matrix dimension')
    plt.savefig('plot/CUDAmisses_plot')

client = MongoClient('mongodb://localhost:27017/')
db = client['AMSC_PROJECT']  # Cambia con il nome del tuo database
collection = db['filresult']
upload_new_data()
CUDAmisses_plot(collection)



