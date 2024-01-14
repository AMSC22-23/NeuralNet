from pymongo import  MongoClient
import pandas as pd
#suppressing all warnings
import warnings
warnings.filterwarnings("ignore")


def upload_new_data():

    client = MongoClient('mongodb://localhost:27017/')
    db = client['AMSC_PROJECT']
    collection = db['filresult']

    # reading the file containing the profiling results, located in a folder above
    csv_file_path = '../profiling_results.csv'

    #csv_file_path = '../profiling_results.csv'
    data = pd.read_csv(csv_file_path, names=['author', 'id', 'matrix_dimension', 'datatype',
                                             'time [ms]', 'tile_dim', 'threads', 'misses'], header=None)

    # forcing the id to be a string
    data['id'] = data['id'].astype(str)

    """
    # Corrects the csv file by swapping the values of misses and tile_dim when misses is NaN

    for i in range(len(data)):
        if pd.isnull(data['misses'][i]):
            data['misses'][i] = data['tile_dim'][i]
            data['tile_dim'][i] = -1
    """

    data_dict = data.to_dict(orient='records')


    # inserting the data in the database
    count_inserted = 0
    for i in range(len(data_dict)):
        try:
            collection.insert_one(data_dict[i])
            count_inserted += 1
        except:
            print("Error in inserting element " + str(i))
            continue


    print("Inserted " + str(count_inserted) + " elements")



    # deleting the content of the csv file
    f = open(csv_file_path, "w+")
    f.close()

    return collection
