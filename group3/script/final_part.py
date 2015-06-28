'''data= 
label= '''
	from sklearn import cross_validation

	#------DECISION TREE : BEGIN

	from sklearn.tree import DecisionTreeClassifier
	clftree = DecisionTreeClassifier() 

	x_train, x_test, l_train, l_test = cross_validation.train_test_split(data, label, test_size=0.2) #ACTUALLY, l_train and l_test are trash: just in case you need to have a single pointage (it's usueless due to the fact we use crs val


	clftree= clftree.fit(x_train, l_train) #"FIT" THE  stuff	
	scores = cross_validation.cross_val_score(clftree,data,label,cv=5) 
	labelsPredict=clftree.predict(data)

	diz = {'STILL':1, 'STILL_ARM':2, 'WALK':3,'RUN':4}

	for i in range(len(labelsPredict)):
		labelsPredict[i]=diz[labelsPredict[i]]
	labelsPredict=np.array(labelsPredict[:].astype(int))

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

	x_train, x_test, l_train, l_test = cross_validation.train_test_split(data, label, test_size=0.2) #ACTUALLY, l_train and l_test are trash: just in case you need to have a single pointage (it's usueless due to the fact we use crs val

	clfsvm= clf.fit(x_train, l_train) #"FIT" THE  stuff
	scores = cross_validation.cross_val_score(clfsvm,data,label,cv=5)
	labelsPredict=clfsvm.predict(data)
	for i in range(len(labelsPredict)):
		labelsPredict[i]=diz[labelsPredict[i]]
	labelsPredict=np.array(labelsPredict[:].astype(int))
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

	x_train, x_test, l_train, l_test = cross_validation.train_test_split(data, label, test_size=0.2) #ACTUALLY, l_train and l_test are trash: just in case you need to have a single pointage (it's usueless due to the fact we use crs val

	clfforest= clf.fit(x_train, l_train) #"FIT" THE  stuff	
	scores = cross_validation.cross_val_score(clfforest,data,label,cv=5) 
	labelsPredict=clfforest.predict(data)
	for i in range(len(labelsPredict)):
		labelsPredict[i]=diz[labelsPredict[i]]
	labelsPredict=np.array(labelsPredict[:].astype(int))
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
	for i in range(len(labelsPredict)):
		labelsPredict[i]=diz[labelsPredict[i]]
	labelsPredict=np.array(labelsPredict[:].astype(int))
	#msg = "the precision of the adaboost  is %6f \n" % (mean(scores)) 
	plt.figure()
	plt.title( 'm.f.:adaboost, prec ='+ str( np.mean(scores) )  )    #??? how to lables and title the plot?
	plt.xlabel('time (0,5s)')
	plt.ylabel('activities')
	plt.plot(range(len(labelsPredict)), labelsPredict)
	#------ADABOOST : END

	import cPickle
	# save the classifier

	with open('tree.pkl', 'wb') as fid:
	    cPickle.dump(clftree, fid)
	with open('svm.pkl', 'wb') as fid:
	    cPickle.dump(clfsvm, fid) 
	with open('forest.pkl', 'wb') as fid:
	    cPickle.dump(clfforest, fid) 
	with open('adaboost.pkl', 'wb') as fid:
	    cPickle.dump(clfadaboost, fid) 
	plt.show()
'''
import pickle
>>> s = pickle.dumps(clftree)
>>> clf2 = pickle.loads(s)'''
#nota: stampare assolutamente il msg da qualche parte!!!!
