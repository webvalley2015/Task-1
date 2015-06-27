import numpy as np
import matplotlib.pyplot as plt

'''
Debugging library to easily make graphs
'''

def importData(text):
	data_all = np.genfromtxt(text,delimiter=',',skip_header=2)
	return data_all

def selectRows(data):
	col0 = data[:,0]
	col0 = col0.reshape((col0.shape[0],1))
	col234 = data[:,2:5]
	return np.concatenate([col0,col234],axis = 1)

def plotData(data):
	#plt.figure()
	plt.plot(data)
	plt.show()

def plot_everything(text):
	a = importData(text)
	a = selectRows(a)
	plotData(a)