from dbconnection import upload_new_data
from pymongo import MongoClient
import pandas as pd
from matplotlib import pyplot as plt
from plot_fromDB import id_to_name



def ploteffect():
    client = upload_new_data()
    data = client.aggregate([
        {
            "$match": {
                "matrix_dimension": "3000X3000",
                "id": {"$in": ["108", "118", "128"]},

                "datatype": "float"
            }
        },
        {
            "$group": {
                "_id": "$id",
                "minTime": {"$min": "$time [ms]"}
            }
        },
        {
            "$sort": {
                "minTime": -1  # -1 indica l'ordine decrescente
            }
        }
    ]
    )

    data = pd.DataFrame(data)
    print(data)



    plt.figure()
    plt.bar("-O3 -ffast-math -march=native", data["minTime"][0])
    plt.bar("previous and -funroll-loops ", data["minTime"][1])
    plt.bar("previous and -ftracer", data["minTime"][2])
    plt.xticks(rotation=5)
    plt.ylabel("Time [ms]")

    plt.title("Effect of different compiler flags on gmultiT on 3000X3000 matrix")
    plt.savefig('../plot/flagEffect')



ploteffect()