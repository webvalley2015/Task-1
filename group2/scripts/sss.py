import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
 

filenames = ['/home/federico/Desktop/data/federico-standing.csv', '/home/federico/Desktop/data/federico-moving-arms.csv', '/home/federico/Desktop/data/federico-walking.csv', '/home/federico/Desktop/data/federico-running.csv']
    
labels = ['01','02','03','04']

#var definition 
data_all = []
data_mean = []          
data_var=[]
data_combined = [0,0,0,0,0,0,0,0,0];
data_mean_combined = [0,0,0,0,0,0,0,0,0];
data_var_combined = [0,0,0,0,0,0,0,0,0];
#data=[]

WINDOW_SIZE = 50
STEP = 25
 
N = len(filenames)

for i in range(N):
	
	# Add data from next file

    # pd.read_table infers the column header and expose the ability to 
    # skip initial rows (i.e. metadata)

    data_all.append(np.loadtxt(filenames(i))

	
    data_combined = np.vstack([data_combined, data_all[i]])

    [row, col] = data_all[i].shape
	
    _mean = np.zeros(( len(range(0, row, STEP)) , col ))
    _var = np.zeros(( len(range(0, row, STEP)) , col ))
    
    for i_col in range(col):
        idx = 0
		
        for i_row in range(0, row, STEP):
	
			# compute mean of window elements
			
            _mean[idx,i_col] = np.mean(data_all[i][i_row:i_row+WINDOW_SIZE-1,i_col])
            _var[idx,i_col] = np.var(data_all[i][i_row:i_row+WINDOW_SIZE-1,i_col])
            idx += 1
    data_mean.append(_mean)
    data_mean_combined = np.vstack([data_mean_combined, data_mean[i]])
	
    data_var.append(_var)
    data_var_combined = np.vstack([data_var_combined, data_var[i]])

np.save('data_mean', data_mean)
np.save('data_var', data_var)


