import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

labels = ["STILL","STILL_ARM","WALK","RUN"]

frequency = 100.0
window_len = 1
window_pass = 0.5

data_labels=["adjsdj"]


def andrewtegify(whatever, start, end):
    duration = start - end
    max = np.amax(whatever) * duration
    min = np.amin(whatever) * duration
    steps = abs(duration / len(whatever))
    axis = np.arange(start, end, steps)
    while len(whatever) != len(axis):
        if len(whatever) > len(axis): 
            #print 'add'
            axis = np.append(axis, (axis[len(axis)-1]+steps))
        elif len(whatever) < len(axis): 
            #print 'remove'
            axis = np.delete(axis, (len(axis) - 1))
        #print str(len(whatever)) + ' ' +  str(len(axis))
        
    totar = 0
    if max < 0 and min < 0:
        va = min - max
        tarea = scipy.integrate.simps(whatever, axis)
        totar = va / tarea
    elif max > 0 and min < 0:
        va = max - min
        tarea = scipy.integrate.simps(whatever, axis)
        totar = va / tarea
        
    elif max > 0 and min > 0:
        va = max - min
        tarea = scipy.integrate.simps(whatever, axis)
        totar = va / tarea
    return totar
    



for i in labels:
	filename = i+".txt"
	part_data=np.genfromtxt(i+".txt", delimiter=',')
	part_data = part_data[:,2:]
	begin = 0
	end = begin+ (window_pass*frequency)
	while(end < part_data.shape[0]):  #mean, std, max/min, medium, norm: FEATURE EXTRACTING
		_avg= part_data[begin:end, :].mean(axis=0)
		_std=np.std(part_data[begin:end, :],axis=0)
 		_integ = andrewtegify(part_data[begin:end], begin, end)
		_min=part_data[begin:end, :].min(axis=0)
		_max=np.max(part_data[begin:end, :],axis=0)
		_medium=np.median(part_data[begin:end, :], axis=0)
		_norm=np.linalg.norm(part_data[begin:end, :], axis=0)	
		
		if(begin==0 and i=="STILL"):
			mean=_avg
			std=_integ
			minimun=_min
			maximum=_max
			medium=_medium
			norm=_norm
			data_labels[0]=i
			
		else:
			mean=np.vstack((mean,_avg))
			std=np.vstack((std,_integ))	
			minimum=np.vstack((minimun,_min))
			maximum=np.vstack((maximum,_max))
			medium=np.vstack((medium,_medium))
			norm=np.vstack((norm,_norm))
			data_labels.append(i)
		
		begin=begin + (window_pass*frequency)
		end = begin + (window_len*frequency)
'''	
	plt.plot(mean)
	plt.title(i)
	plt.show()
'''

	
#MEAN OUTPUT
out_mean=np.array(["","","",""])	

for i in range(mean.shape[0]):
	if (i==0):
		out_mean=np.array([ data_labels[i], str(mean[i,0]),str(mean[i,1]),str(mean[i,2]) ])
	else:
		out_mean=np.vstack((out_mean,[  data_labels[i], str(mean[i,0]),str(mean[i,1]),str(mean[i,2]) ]))

#np.save("mean",out_mean)

#STD OUTPUT

out_std=np.array(["","","",""])	

for i in range(mean.shape[0]):
	if (i==0):
		out_std=np.array([ data_labels[i], str(mean[i,0]),str(mean[i,1]),str(mean[i,2]) ])
	else:
		out_std=np.vstack((out_std,[  data_labels[i], str(mean[i,0]),str(mean[i,1]),str(mean[i,2]) ]))

#np.save("std",out_std)

#MAX OUTPUT

out_max=np.array(["","","",""])	

for i in range(mean.shape[0]):
	if (i==0):
		out_max=np.array([ data_labels[i], str(mean[i,0]),str(mean[i,1]),str(mean[i,2]) ])
	else:
		out_max=np.vstack((out_max,[  data_labels[i], str(mean[i,0]),str(mean[i,1]),str(mean[i,2]) ]))

#np.save("max",out_max)
	

#MINIMUM OUTPUT
out_min=np.array(["","","",""])	

for i in range(mean.shape[0]):
	if (i==0):
		out_min=np.array([ data_labels[i], str(mean[i,0]),str(mean[i,1]),str(mean[i,2]) ])
	else:
		out_min=np.vstack((out_min,[  data_labels[i], str(mean[i,0]),str(mean[i,1]),str(mean[i,2]) ]))

#np.save("min",out_min)	

#MEDIUM OUTPUT


out_medium=np.array(["","","",""])	

for i in range(mean.shape[0]):
	if (i==0):
		out_medium=np.array([ data_labels[i], str(mean[i,0]),str(mean[i,1]),str(mean[i,2]) ])
	else:
		out_medium=np.vstack((out_medium,[  data_labels[i], str(mean[i,0]),str(mean[i,1]),str(mean[i,2]) ]))

#np.save("mean",out_medium)


#NORM OUTPUT

out_norm=np.array(["","","",""])	

for i in range(mean.shape[0]):
	if (i==0):
		out_norm=np.array([ data_labels[i], str(mean[i,0]),str(mean[i,1]),str(mean[i,2]) ])
	else:
		out_norm=np.vstack((out_norm,[  data_labels[i], str(mean[i,0]),str(mean[i,1]),str(mean[i,2]) ]))

#np.save("norm",out_norm)

	
#--------"second" part=
label=np.array(out_mean[:,0])
mean = np.array(out_mean[:,1:].astype(float))
std=np.array(out_std[:,1:].astype(float))
norm=np.array(out_norm[:,1:].astype(float))
data=np.hstack((mean,std,norm))


from sklearn import cross_validation
from sklearn.tree import DecisionTreeClassifier

x_train, x_test, l_train, l_test = cross_validation.train_test_split(data, label, test_size=0.2)

#from here now, we will build the model for selection

clf = DecisionTreeClassifier() #!!!!!!!()!!!!!

clf_fitted= clf.fit(x_train, l_train) #"FIT" THE  stuff	
#cross_validation.cross_val_score(clf_fitted,data,label,cv=5) F**K THIS					 **** !!!


'''
#WE DID THE CLF NOW. I THINK IT'S EQUAL TOUGH
#SVM!
from sklearn import svm
clf=svm.LinearSVC(C=1)
clf_fitted=clf.fit(x_train,l_train)

#other methods??
'''

labelsPredict=clf_fitted.predict(data)
plt.figure()
diz = {'STILL':1, 'STILL_ARM':2, 'WALK':3,'RUN':4}

for i in range(len(labelsPredict)):
	labelsPredict[i]=diz[labelsPredict[i]]
	#l_test[i]=diz[l_test[i]]	

labelsPredict=np.array(labelsPredict[:].astype(int))

plt.plot(range(len(labelsPredict)), labelsPredict)
plt.show()
#-----------------------------------------------------

'''
#ok, now plot this shit
clf = clf.fit(dataTrain, labelsTrain)
labelsPredict = clf.predict(dataTest)
score = clf.score(dataTest, labelsTest)


print 'Classification Accuracy %f' % score

plt.figure()
plt.plot(range(len(labelsTest)), labelsTest, range(len(labelsPredict)), labelsPredict)




labelsPredict=clf.predict(x_test)
plt.figure()
dict = {'LAYING':1, 'SITTING':2, 'STANDING':3,'WALKING':4,'WALKUPS':5,'WALKDWN':6}
dict2 = {'1':1, '2':2, '3':3, '4':4, '5':5, '6':6}
for i in range(len(labelsPredict)):
	labelsPredict[i]=dict2[labelsPredict[i]]
	l_test[i]=dict[l_test[i]]	

plt.plot(range(len(data)), data, range(len(labelsPredict)), labelsPredict)
plt.show()
#---------------

'''

