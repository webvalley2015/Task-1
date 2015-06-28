__author__ = 'federico'

import pandas as pd
import numpy as np
import MySQLdb

def get_raw_data_from_db():
    '''
    Description: get raw data from db and return an array
    Argoments: NONE
    Returns: raw_data_array
    '''

    db = MySQLdb.connect(host="localhost", 
                         user="root", 
                          passwd="###", 
                          db="Group2") 
    cur = db.cursor()

    cur.execute("SELECT * FROM raw_data")

    result = cur.fetchall()

    cur.close()
    db.close()

    return result[0]

def get_elaborated_data_from_db():
    '''
    Description: get elaborated data from db and return an array
    Argoments: NONE
    Returns: elaborated_data_array
    '''

    db = MySQLdb.connect(host="localhost", 
                         user="root", 
                          passwd="###", 
                          db="Group2") 
    cur = db.cursor()

    cur.execute("SELECT * FROM elaborated_data")

    result = cur.fetchall()

    cur.close()
    db.close()

    return result[0]

def put_elaborated_data_in_db(elaborated_dataframe):
    '''
    Description: put elaborated data in db
    Argoments: elaborated_data_array
    Returns: NONE
    '''

    db = MySQLdb.connect(host="localhost", 
                         user="root", 
                          passwd="###", 
                          db="Group2") 

    rows_numb = elaborated_dataframe.shape[0]

    cur = db.cursor()

    for i in range(0,rows_numb):
        acc_mean = float(elaborated_dataframe[i][0])
        gyr_mean = float(elaborated_dataframe[i][1])
        mag_mean = float(elaborated_dataframe[i][2])
        action = "0"
        cur.execute("""INSERT INTO elaborated_data (AccMean, GyrMean, MagMean, action) VALUES (%s, %s, %s, %s);""",
                (acc_mean, gyr_mean, mag_mean, action))
        db.commit()
    cur.close()
    db.close()
