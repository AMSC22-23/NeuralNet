from pymongo import MongoClient
import pandas as pd

client = MongoClient('mongodb://localhost:27017/')
db = client['AMSC_PROJECT']
collection = db['filresult']

csv_file_path = '../filResult.csv'
data = pd.read_csv(csv_file_path, names=['author', 'id', 'matrix_dimension', 'datatype',
                                         'time [ms]', 'tile_dim', 'threads', 'misses'], header=None)


data_dict = data.to_dict(orient='records')
data['id'] = data['id'].astype(str)

count_inserted = 0
for i in range(len(data_dict)):
    try:
        collection.insert_one(data_dict[i])
        count_inserted += 1
    except:
        print("Error in inserting element " + str(i))
        continue


print("Inserted " + str(count_inserted) + " elements")
