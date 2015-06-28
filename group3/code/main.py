#import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import feat_script as f

#define costants
labels = {'still':0,
        'arm':1,
        'walk':2,
        'run':3
        }

#import, preprocessing and windowing (with feat extraction)

ll=[]
ok_data = pd.DataFrame()
curr_graph = 0
for i in labels:
    filename = './data/'+i+'.txt'
    raw_data=np.genfromtxt(filename, delimiter=',')
    raw_data = raw_data[:,2:]
    #f.raw_plot(raw_data)
    ll.append(raw_data)
    exf_data = f.windowAnalize(raw_data, labels[i])
    ok_data = ok_data.append(exf_data)
    
ok_data.to_csv(path_or_buf='./Group2_extracted')

#f.feat_barplot(ok_data)
    
#pre_data = ok_data.copy()
#sol = pre_data[pre_data.columns[len(pre_data.columns)-1]]
#in_data = pre_data[pre_data.columns[:-1]]

#print f.cvDCT(in_data, sol)
#print f.cvSVC(in_data, sol)
