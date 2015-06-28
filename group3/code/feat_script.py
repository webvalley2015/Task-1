import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm

WINLEN = 100
WINSTEP = 50

cols = ["AccX","AccY","AccZ", "GyrX","GyrY","GyrZ", "MagX","MagY","MagZ"]

colnames = ['Acc.std.Norm', 'Mag.std.Norm', 'Mag.Mean.Z', 'Mag.Mean.X', 'Mag.Mean.Y', 'Acc.std.Y', 'Acc.std.X', 'Acc.std.Z', 'Acc.Max.X', 'label', 'Gyr.Min.Z', 'Gyr.Min.Y', 'Gyr.Min.X', 'Mag.Min.Norm', 'Acc.Min.Norm', 'Gyr.Min.Norm', 'Mag.Mean.Norm', 'Acc.Mean.X', 'Acc.Mean.Y', 'Acc.Mean.Z', 'Gyr.Max.Norm', 'Mag.Max.Norm', 'Acc.Max.Norm', 'Gyr.Mean.Norm', 'Acc.Max.Y', 'Acc.Max.Z', 'Gyr.Mean.Z', 'Gyr.Mean.Y', 'Gyr.Mean.X', 'Mag.Max.Z', 'Mag.Max.X', 'Mag.Max.Y', 'Acc.Mean.Norm', 'Acc.Min.Z', 'Acc.Min.X', 'Acc.Min.Y', 'Gyr.std.Norm', 'Mag.Min.X', 'Mag.Min.Y', 'Mag.Min.Z', 'Mag.std.Z', 'Mag.std.Y', 'Mag.std.X', 'Gyr.std.X', 'Gyr.std.Y', 'Gyr.std.Z', 'Gyr.Max.Y', 'Gyr.Max.X', 'Gyr.Max.Z']

def exFeat(l_curr, lab):
    mean = l_curr.mean()
    std = l_curr.std()
    mi = l_curr.min()
    ma = l_curr.max()
    feat = {
    'Acc.Mean.X': mean.AccX, 
    'Acc.Mean.Y': mean.AccY,
    'Acc.Mean.Z': mean.AccZ,
    'Acc.Mean.Norm': mean.AccNorm,
    'Gyr.Mean.X': mean.GyrX, 
    'Gyr.Mean.Y': mean.GyrY,
    'Gyr.Mean.Z': mean.GyrZ,
    'Gyr.Mean.Norm': mean.GyrNorm,
    'Mag.Mean.X': mean.MagX, 
    'Mag.Mean.Y': mean.MagY,
    'Mag.Mean.Z': mean.MagZ,
    'Mag.Mean.Norm': mean.MagNorm,
    'Acc.Max.X': ma.AccX, 
    'Acc.Max.Y': ma.AccY,
    'Acc.Max.Z': ma.AccZ,
    'Acc.Max.Norm': ma.AccNorm,
    'Gyr.Max.X': ma.GyrX, 
    'Gyr.Max.Y': ma.GyrY,
    'Gyr.Max.Z': ma.GyrZ,
    'Gyr.Max.Norm': ma.GyrNorm,
    'Mag.Max.X': ma.MagX, 
    'Mag.Max.Y': ma.MagY,
    'Mag.Max.Z': ma.MagZ,
    'Mag.Max.Norm': ma.MagNorm,
    'Acc.Min.X': mi.AccX, 
    'Acc.Min.Y': mi.AccY,
    'Acc.Min.Z': mi.AccZ,
    'Acc.Min.Norm': mi.AccNorm,
    'Gyr.Min.X': mi.GyrX, 
    'Gyr.Min.Y': mi.GyrY,
    'Gyr.Min.Z': mi.GyrZ,
    'Gyr.Min.Norm': mi.GyrNorm,
    'Mag.Min.X': mi.MagX, 
    'Mag.Min.Y': mi.MagY,
    'Mag.Min.Z': mi.MagZ,
    'Mag.Min.Norm': mi.MagNorm,
    'Acc.std.X': std.AccX, 
    'Acc.std.Y': std.AccY,
    'Acc.std.Z': std.AccZ,
    'Acc.std.Norm': std.AccNorm,
    'Gyr.std.X': std.GyrX, 
    'Gyr.std.Y': std.GyrY,
    'Gyr.std.Z': std.GyrZ,
    'Gyr.std.Norm': std.GyrNorm,
    'Mag.std.X': std.MagX, 
    'Mag.std.Y': std.MagY,
    'Mag.std.Z': std.MagZ,
    'Mag.std.Norm': std.MagNorm,
    'label': lab
    }
    #print feat.keys()
    return feat

def windowAnalize (nparr, lab):

    df = pd.DataFrame(nparr, columns = cols)
    AccNorm = np.sqrt(df.AccX**2 + df.AccY**2 + df.AccZ**2)
    GyrNorm = np.sqrt(df.GyrX**2 + df.GyrY**2 + df.GyrZ**2)
    MagNorm = np.sqrt(df.MagX**2 + df.MagY**2 + df.MagZ**2)
    df = pd.concat([df, AccNorm, GyrNorm, MagNorm], axis=1)
    df.columns = ["AccX","AccY","AccZ", "GyrX","GyrY","GyrZ", "MagX","MagY","MagZ","AccNorm","GyrNorm","MagNorm"]
    print(df)
    t_start = df.index[0]
    t_end = t_start + WINLEN
    
    res = pd.DataFrame(columns=colnames)
    while (t_end < df.shape[0]-1):
        df_curr = df.query(str(t_start)+'<=index<'+str(t_end))
        v = exFeat(df_curr, lab)
        newrow = pd.DataFrame(v, index=[t_start])
        res = res.append(newrow)
        #update values
        t_start = t_start + WINSTEP
        t_end = t_start + WINLEN

    return res
    
def raw_plot(data):
    plt.style.use('ggplot')
    plt.figure()
    plt.plot(data[:,list((0,1,2))])
    plt.plot(data[:,list((3,4,5))])
    plt.plot(data[:,list((6,7,8))])
    plt.show()

    #to revise it!!
def feat_barplot(x):
    plt.style.use('ggplot')
    data = pd.DataFrame()
    for i in range(len(x.columns)-1):
        data0 = x.query('label == '+str(0)).iloc[:,i]
        data1 = x.query('label == '+str(1)).iloc[:,i]
        data2 = x.query('label == '+str(2)).iloc[:,i]
        data3 = x.query('label == '+str(3)).iloc[:,i]
        data = [data0, data1, data2, data3]
        plt.subplot(5,10, i+1)
        plt.boxplot(data)
        plt.title(x.columns[i])
    plt.show()
    
    
def cvDCT(raw_df, sol):
    x = raw_df
    y = sol
    #x_train, x_test, y_train, y_test = cross_validation.train_test_split(x,y, test_size=0.2)
    clf = DecisionTreeClassifier()
    scores = cross_validation.cross_val_score(clf, x, y, cv=5)
    accuracy = np.array((scores.mean(), scores.std()*2))
    return accuracy
    
def cvSVC(raw_df, sol):
    x = raw_df
    y = sol
    #x_train, x_test, y_train, y_test = cross_validation.train_test_split(x,y, test_size=0.4)
    Clist = [ 10**i for i in range(-5,8) ]
    accuracy = np.zeros(3)#.reshape((1,-1))
    for par in Clist:
        clf = svm.LinearSVC(C=par)
        #clf = clf.fit(x_train, y_train)
        scores = cross_validation.cross_val_score(clf, x, y, cv=5)
        print(scores)
        vec = np.array([par, scores.mean(), scores.std()*2])
        #print vec
        accuracy = np.concatenate((accuracy, vec))
        
    accuracy = accuracy.reshape((-1,3))
    accuracy = accuracy[1:,:]
    plt.figure()
    plt.plot(accuracy[:,0],accuracy[:,1])
    plt.xscale('log')
    plt.show()    
    return accuracy
    
    

