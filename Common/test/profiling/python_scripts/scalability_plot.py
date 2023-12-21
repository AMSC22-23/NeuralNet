from dbconnection import upload_new_data
from plot_fromDB import id_to_name
from pymongo import MongoClient
import pandas as pd
import matplotlib.pyplot as plt
def scalability_plot(collection):

    # we query all entries where id = 128 and threads is a power of 2

    query = {"id": "128", "threads": {"$mod": [2, 0]}, "matrix_dimension": "4000X4000"}



    gmultiT = collection.find(query)

    # convert to dataframe
    gmultiT = pd.DataFrame(list(gmultiT))

    # we consider duplicates all the entries with the same number of threads
    # for each duplicate we take the one that has the lowest time

    gmultiT = gmultiT.sort_values(by=['time [ms]'])
    gmultiT = gmultiT.drop_duplicates(subset=['threads'])

    # sorting by number of threads
    gmultiT = gmultiT.sort_values(by=['threads'])

     # plotting
    plt.plot(gmultiT['threads'], gmultiT['time [ms]'])
    plt.xlabel('# threads')
    plt.ylabel('time [ms]')
    plt.grid()
    plt.title('Execution time vs # threads')
    plt.savefig('plot/scalability_plot')

    plt.close()




upload_new_data()


# we now connect to the database
client = MongoClient('mongodb://localhost:27017/')
db = client['AMSC_PROJECT']  # Cambia con il nome del tuo database
collection = db['filresult']

scalability_plot(collection)