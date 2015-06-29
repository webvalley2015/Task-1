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
    plt.subplot(3,4,i+1)
    i += 1
    plt.plot(data[:,[0,1,2]])
    plt.plot(data[:,[3,4,5]])
    plt.plot(data[:,[6,7,8]])
    #plt.title(labels)
    plt.plot()

    
def feat_barplot(x):
    plt.style.use('ggplot')
    data = pd.DataFrame()
    for i in range(len(x.columns)-1):
    #    curr[i] = x[x.label == i]
        data0 = x.query('label == '+str(0)).iloc[:,i]
        data1 = x.query('label == '+str(1)).iloc[:,i]
        data2 = x.query('label == '+str(2)).iloc[:,i]
        data3 = x.query('label == '+str(3)).iloc[:,i]
        data = [data0, data1, data2, data3]
        plt.subplot(4,4, i+1)
        plt.boxplot(data)
        #plt.xaxis.set_visible(False)
        #plt.yticks('False')
        plt.title(x.columns[i])

    plt.show()    
    
def cvDCT(raw_df, sol):
    x = raw_df
    y = sol
    x_train, x_test, y_train, y_test = cross_validation.train_test_split(x,y, test_size=0.2)
    clf = DecisionTreeClassifier()
    scores = cross_validation.cross_val_score(clf, x_test, y_test, cv=15)
    accuracy = np.array((scores.mean(), scores.std()*2))
    return accuracy
    
    #--------------__!_!_!_!_!_!_!-!-!-!-!-!-!-!-!-!-!-!_!_

def justFitEvery1 (label, data):
	#this function takes in input a vector (numpy array) of labels (label) of dim. [n] where n is the number of processed (which means winodwed) istances (so, of course, label[i] contains the REAL label that corresponds to the activity the subject was doing at the x-th istance of data. The x-th istance of data can be translated in time with this equation: Time= x * 0,5  (s))
	#the other vector (numpy array) this function takes in, data, is of the dimension (n,x) where n is the number of processed (which means winodwed) istances (see also comment before ) and x is the number of features we extracted

	#this function output 8 things: for each (of the four) predictive model SVM, Decision tree, Random forest and adaboost ouptuts a file (.pkl) which contains the model, and a graph which shows the predictions of the label made with the given model, for each n (see comment before) istance of the processed data
	 
	from sklearn import cross_validation

	label = np.array(label)
	data = np.array(data)	
	#------DECISION TREE : BEGIN
	
	from sklearn.tree import DecisionTreeClassifier
	clftree = DecisionTreeClassifier() 

	x_train, x_test, l_train, l_test = cross_validation.train_test_split(data, label, test_size=0.2) #ACTUALLY, l_train and l_test are trash: just in case you need to have a single pointage (it's usueless due to the fact we use crs val


	clftree= clftree.fit(x_train, l_train) #"FIT" THE  stuff	
	scores = cross_validation.cross_val_score(clftree,data,label,cv=5) 
	labelsPredict=clftree.predict(data)

	'''diz = {'STILL':1, 'STILL_ARM':2, 'WALK':3,'RUN':4}

	for i in range(len(labelsPredict)):
		labelsPredict[i]=diz[labelsPredict[i]]
	labelsPredict=np.array(labelsPredict[:].astype(int))'''

	plt.plot(range(len(labelsPredict)), labelsPredict)

	#msg = "the precision of the Decision tree is %6f \n" % (np.mean(scores)) 
	#print msg
	plt.figure()
	plt.title( 'm.f. : decision tree, prec ='+ str( np.mean(scores) )  )   #??? how to lables and title the plot?
	plt.xlabel('time (x*0,5s)')
	plt.ylabel('activities')
	plt.plot(range(len(labelsPredict)), labelsPredict)
	#-----DECISION TREE: END

	#------ SVM  BEGIN
	from sklearn import svm

	clf = svm.SVC( C=10)

	x_train, x_test, l_train, l_test = cross_validation.train_test_split(data, label, test_size=0.2) #ACTUALLY, l_test and x_test are trash: just in case you need to have a single pointage (it's usueless due to the fact we use crs val

	clfsvm= clf.fit(x_train, l_train) #"FIT" THE  stuff
	scores = cross_validation.cross_val_score(clfsvm,data,label,cv=5)
	labelsPredict=clfsvm.predict(data)
	
	'''for i in range(len(labelsPredict)):
		labelsPredict[i]=diz[labelsPredict[i]]
	labelsPredict=np.array(labelsPredict[:].astype(int))'''
	#msg = "the precision of the support vector machine is %6f \n " % (np.mean(scores))
	#print msg
	plt.figure()
	plt.title( 'm.f.:svm, prec ='+ str( np.mean(scores) )  )    #??? how to lables and title the plot?
	plt.xlabel('time (0,5s)')
	plt.ylabel('activities')
	plt.plot(range(len(labelsPredict)), labelsPredict)
	#-------- SVM : END


	#-------- RANDOM FOREST : BEGIN
	from sklearn.ensemble import RandomForestClassifier
	clf = RandomForestClassifier()

	x_train, x_test, l_train, l_test = cross_validation.train_test_split(data, label, test_size=0.2) #ACTUALLY, l_test and x_test are trash: just in case you need to have a single pointage (it's usueless due to the fact we use crs val

	clfforest= clf.fit(x_train, l_train) #"FIT" THE  stuff	
	scores = cross_validation.cross_val_score(clfforest,data,label,cv=5) 
	labelsPredict=clfforest.predict(data)
	'''for i in range(len(labelsPredict)):
		labelsPredict[i]=diz[labelsPredict[i]]
	labelsPredict=np.array(labelsPredict[:].astype(int))'''
	#msg = "the precision of the random forest  is %6f \n" % (np.mean(scores)) 
	#print msg
	plt.figure()
	plt.title( 'm.f.:random forest, prec ='+ str( np.mean(scores) )  )    #??? how to lables and title the plot?
	plt.xlabel('time (0,5s)')
	plt.ylabel('activities')
	plt.plot(range(len(labelsPredict)), labelsPredict)
	#RANDOM FOREST :END



	#------ADABOOST : BEGIN
	from sklearn.ensemble import AdaBoostClassifier
	clf = AdaBoostClassifier(n_estimators=100)

	x_train, x_test, l_train, l_test = cross_validation.train_test_split(data, label, test_size=0.2) #ACTUALLY, l_train and l_test are trash: just in case you need to have a single pointage (it's usueless due to the fact we use crs val

	clfadaboost= clf.fit(x_train, l_train) #"FIT" THE  stuff	
	scores = cross_validation.cross_val_score(clfadaboost,data,label,cv=5) 
	labelsPredict=clfadaboost.predict(data)
	'''for i in range(len(labelsPredict)):
		labelsPredict[i]=diz[labelsPredict[i]]
	labelsPredict=np.array(labelsPredict[:].astype(int))'''
	#msg = "the precision of the adaboost  is %6f \n" % (mean(scores)) 
	plt.figure()
	plt.title( 'm.f.:adaboost, prec ='+ str( np.mean(scores) )  )    #??? how to lables and title the plot?
	plt.xlabel('time (0,5s)')
	plt.ylabel('activities')
	plt.plot(range(len(labelsPredict)), labelsPredict)
	#------ADABOOST : END

	import cPickle
	# save the classifier

	with open('decision_tree.pkl', 'wb') as fid:
	    cPickle.dump(clftree, fid)
	with open('svm.pkl', 'wb') as fid:
	    cPickle.dump(clfsvm, fid) 
	with open('random_forest.pkl', 'wb') as fid:
	    cPickle.dump(clfforest, fid) 
	with open('adaboost.pkl', 'wb') as fid:
	    cPickle.dump(clfadaboost, fid) 
	plt.show()
 	
    

def predict_plot(label, data):	
#this func. predict the data given in (for the format of data, see the prevouis function, second comment) and plots the label prediction for each istance of the data.

#the previous comment explained the main objective of this functuon. Due to educational aims, this functions picks also the label array in input, so that she can check with the cross validation the actual accuracy.

#ps: thinking about a real use (on a phone mabye?) it's obvious we wouln't get labelled data. but since here we actually get labelled data, we can use those labels. 
	label = np.array(label)
	data = np.array(data)		
	
	data = np.array(data)	
	file_list=['decision_tree','svm','random_forest','adaboost']	
	import cPickle
	for i in file_list:
		with open(i+'.pkl', 'rb') as fid:
   			 clf = cPickle.load(fid)
		labelsPredict=np.array(data[:].astype(int))
		labelsPredict=clf.predict(data)
		#msg = "the precision of the adaboost  is %6f \n" % (mean(scores)) 
		sco = cross_validation.cross_val_score(clf,data,label,cv=5) 		
		sco = np.mean(sco)	
		sco = 100*sco
		sco = "%2f" % (sco)	
		plt.figure()
		plt.title(i+'  , precision '+ sco)    #??? how to lables and title the plot?
		plt.xlabel('time (0,5s)')
		plt.ylabel('activities')
		plt.plot(range(len(labelsPredict)), labelsPredict)
		
	plt.show()	   
	    
'''
here i try to explain what does n, x , and processed data means, and how to relate them with the REAL TIME

pose we have a sampling frequency of 100 Hz, and for example, 3000 RAW collected data
when you process it with the windowing, you reduce the number of istantces by a factor equal, more or less, to the window shift (the more or less is related to the fact that the lenght of the raw data is not multiple of the window shift, but actually, in the worst cae, you can lost 0,99s of recordings for each file)

so when i said n i intended the number of processed data (of windows)

with the vector label[] containing n istances, --> time(i) = i * window shift , for each i in range [0,n)  
'''   
    
