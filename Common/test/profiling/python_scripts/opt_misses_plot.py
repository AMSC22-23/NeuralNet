from dbconnection import upload_new_data
from plot_fromDB import id_to_name
from pymongo import MongoClient
import pandas as pd
import matplotlib.pyplot as plt

def opt_misses_plot(collection):


    matrix_size = "512X512"


    ids = ["151", "161", "171", "181"]

    #query all entries where id is in ids and matrix_dimension = matrix_size
    query = {"id": {"$in": ids}, "matrix_dimension": matrix_size}

    data = collection.find(query)

    # convert to dataframe
    data = pd.DataFrame(list(data))

    plt.bar(data['id'], data['misses'])
    plt.xlabel('id')
    plt.ylabel('misses')

    plt.title('Optimized misses vs id')

    plt.savefig('plot/opt_misses_plot')


client = MongoClient('mongodb://localhost:27017/')
db = client['AMSC_PROJECT']  # Cambia con il nome del tuo database
collection = db['filresult']
upload_new_data()

opt_misses_plot(collection)