#import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import feat_script as f


#define costants
labels = {'STILL':0,
        'STILL_ARM':1,
        'WALK':2,
        'RUN':3
        }

#import, preprocessing and windowing (with feat extraction)

ll=[]
ok_data = pd.DataFrame()
curr_graph = 0
for i in labels:
    filename = '../data/'+i+'.txt'
    raw_data=np.genfromtxt(filename, delimiter=',')
    raw_data = raw_data[:,2:]
    #f.raw_plot(raw_data, curr_graph)
    ll.append(raw_data)
    exf_data = f.windowAnalize(raw_data, labels[i])
    ok_data = ok_data.append(exf_data)
    
#ok_data.to_csv(path_or_buf='./Group3_extracted')
    
pre_data = ok_data.copy()
sol = pre_data[pre_data.columns[len(pre_data.columns)-1]]
in_data = pre_data[pre_data.columns[:-1]]



print "do you want to fit the model or to predict some data? (just PLEASE press 'f' OR 'p')"
choiche = input()

if (choiche=='f'):
#--------------FIT!!!!-----------
	f.justFitEvery1(sol, in_data)
#---------------PREDICT
else:
	f.predict_plot(sol, in_data)	



#return ((clftree, clfsvm, clfforest, clfadaboost))
