from dbconnection import upload_new_data
from pymongo import MongoClient
import pandas as pd
from matplotlib import pyplot as plt
from plot_fromDB import id_to_name

def plot_time_complexity(datatype: str):
   plt.figure()

   collection = upload_new_data()

   for id in ['121', '122', '125', '126', '127']:
        data = collection.aggregate([
           {
           '$match': {
               'id': id,
               'datatype': datatype,
               "time [ms]": { '$gt': 0 }
                }
           },
           {
            '$group': {
                '_id': "$matrix_dimension",
                'minTime': { '$min': "$time [ms]" },
                }
            }
        ])
        data = pd.DataFrame(data)
        data['_id'] = data['_id'].apply(lambda x: int(x.split('X')[0]))
        if id!='126' and id !='127' or datatype == "double":
            data = data[data['_id'] <= 2048]
        else:
            data = data[data['_id'] <= 1024]
        data = data.sort_values(by=['_id'])
        plt.plot(data['_id'], data['minTime'], label=id_to_name([id]))


   for id in ['123']:
       data = collection.aggregate([
           {
               '$match': {
                   'id': id,
                   'datatype': datatype,
                   "time [ms]": { '$gt': 0 },
                   'tile_dim': 32
               }
           },
           {
               '$group': {
                   '_id': "$matrix_dimension",
                   'minTime': { '$min': "$time [ms]" }
               }
           }
       ])

   data = pd.DataFrame(data)
   print(data)
   data['_id'] = data['_id'].apply(lambda x: int(x.split('X')[0]))
   data = data[data['_id'] <= 2048]
   data = data.sort_values(by=['_id'])
   plt.plot(data['_id'], data['minTime'], label=id_to_name([id]))

   for id in ['128']:
       matrix_dims = ['256X256', '512X512', '1024X1024', '2048X2048']
       data = collection.find({'id': id, 'datatype': datatype, 'matrix_dimension': {'$in': matrix_dims}, 'threads': 64})
       data = pd.DataFrame(data)

       print(data)
       data['time [ms]'] = data['time [ms]'].astype(float)
       data['matrix_dimension'] = data['matrix_dimension'].apply(lambda x: int(x.split('X')[0]))
       # removing from data entries with matrix dimension equal greater then 2048
       data = data[data['matrix_dimension'] <= 2048]
       data = data.sort_values(by=['matrix_dimension'])
       data = data.drop_duplicates(subset='matrix_dimension', keep='first')

       plt.plot(data['matrix_dimension'], data['time [ms]'], label=id_to_name([id]))



   plt.yscale('log')
   plt.legend()
   plt.grid()
   plt.title('Time Complexity for ' + datatype)
   plt.savefig('../plot/time_complexity_' + datatype)

plot_time_complexity("float")
plot_time_complexity("double")