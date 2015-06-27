import numpy as np
import matplotlib as plt
from training import *
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation

def import_data (sources):
    '''
    Loads data for source and returns data read
    '''
    data_all=[]                                                     #Returning variable
    for name in sources:                                            #For every file
        data_t=np.genfromtxt(name,delimiter=',',skip_header=2)      #Reads the $name file
    	data_all.append(data_t[:,2:5])                              #Appends the columns we want (accelerometer)
    return data_all

'''
We made this functions to try to lower the noise but we saw that it wasn't worth it since it
lowers the accuracy of our model

def pre_processing(raw_data):
    #Pre-processes data with the selected parameters and returns them
    filtered=[]
    for array in raw_data:
        #array=(8*9.81/32768)*array
        #x = array[:,0]
        #y = array[:,1]
        #z = butter_lowpass_filter(array[:,2], 25, 100, 3)
        #x.shape =[x.shape[0], 1]
        #y.shape =[y.shape[0], 1]
        #z.shape =[z.shape[0], 1]
        #partial=np.concatenate([x, y, z], axis=1)
        #filtered.append(partial)
        filtered.append(butter_lowpass_filter(array, 25, 100, 3))
    return filtered

def butter_lowpass(cutoff, fs, order):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order):
    b, a = butter_lowpass(cutoff, fs, order)
    y = lfilter(b, a, data)
    return y
'''

def extract_info(pre_processed_data):
    '''
    Computes something to the data to extract what we want (in this example the norm)
    '''
    data_all=[]                                                     #Returning variable
    for activity_data in pre_processed_data:                        #For every activity array
        x=activity_data[:,0]                                        #X axis
        y=activity_data[:,1]                                        #Y axis
        z=activity_data[:,2]                                        #Z axis
        norm = np.sqrt(x**2+y**2+z**2)                              #norm
        x.shape = [x.shape[0], 1]                                   #Reshaping for concatenating (he has to have 1 dimension)
        y.shape = [x.shape[0], 1]                                   # "
        z.shape = [x.shape[0], 1]                                   # "
        norm.shape = [norm.shape[0], 1]                             # "
        data_all.append(np.concatenate((x, y, z, norm), axis=1))    #concatenates to make a matrix with 4 columns
    return data_all

def compute_windows(data_length, wlen=100, wstep=50):
    '''
    Returns the list of window starts.
    Gets lenght of signal, window length, window step
    '''
    return range(0, data_length-wlen, wstep)                        #Windows start is just a range

def extract_features(data, wlen, wstep):
    '''
    Computes all the features, given data, window's length and step
    '''
    features_all=[]                                                 #returning variable
    for activity in data:                                           #for every array in data that holds the data for an activity
        windows_list = compute_windows(activity.shape[0], wlen, wstep)
        features_activity=[]                                        #temporary features-holding list to be concatenated
        for wstart in windows_list:                                 #for every window
            wend=wstart+wstep                                       #computes the end of the window
            data_portion=activity[wstart : wend, :]                 #window
            features_portion = get_features_row(data_portion)       #gets a row of features
            features_activity.append(features_portion)              #add new features to features_all
        features_all.append(np.array(features_activity))            #Append the activity-related array
    return features_all

def get_features_row(data_portion, mean=True, std=True, min=False, max=False, ran=True, rms=True):
    '''
    Computes the feature for the given data_portion.
    Modify the parameters to include or exclude some features
    '''
    features=[]                                                     #Features list - RETURN
    for idx in range(data_portion.shape[1]):                        #For every column
        axis=data_portion[:,idx]                                    #Temporary column holding variable
        if mean :                                                   #If $feature adds the feature to the row
            features.append(np.mean(axis))
        if std :
            features.append(np.std(axis))
        if min :
            features.append(np.min(axis))
        if max :
            features.append(np.max(axis))
        if ran :
            features.append(np.max(axis)-np.min(axis))
        if rms :
            features.append(np.sqrt(np.mean(axis**2)))
    return np.array(features)                                       #Returns the row of the features

def get_XY(data):
    '''
    Returns X and Y arrays from $data list of arrays
    '''
    data_types=[]                                               #list holding the arrays types [[0,0,...],...]
    for i, array in enumerate(data):                            #for loop for making a Y array for our machine learning algorithm
        length=array.shape[0]	                                #number of rows
        data_types.append(np.ones(length)*i)                    #Add to data_types an array containing the ID of the activity (i)...
                                                                #...for the length so that it can be used as Y
    Y=np.concatenate(data_types)                                #Builds X array
    X=np.concatenate(data)                                      #Builds Y array
    return (X, Y)

def load_data_from_filelist(filenames):
    '''
    Computes every task from raw_data to machine-learning-ready lists
    '''
    raw_data = import_data(filenames)                           #load data from file list
    #pre_processed_data = pre_processing(raw_data)              #just old code
    #plotter.plotData(raw_data[1])                              #  "
    #plotter.plotData(pre_processed_data[1])                    #  "
    #useful_data = extract_info(pre_processed_data)             #  "
    useful_data =extract_info(raw_data)                         # extract info(norm) from data, we will now have x, y, z, norm columns
    WINLEN=100                                                  #Windows length
    WINSTEP=50                                                  #Windows step
    data_ready = extract_features(useful_data, WINLEN, WINSTEP) #Extract features from data
    return get_XY(data_ready)                                   #Returns X and Y


#List of every group PATH/TO/FILES
groupA= ["A/still.txt", "A/arm.txt", "A/walk.txt", "A/run.txt"]
groupB=["B/FedericoWalkingRawData.txt", "B/FedericoMovingArmRawData.txt", "B/FedericoWalkingRawData.txt", "B/FedericoRunningRawData.txt"]
groupC=["C/Ema_STILL.txt", "C/Ema_STILL_ARM.txt", "C/Ema_WALK.txt", "C/Ema_RUN.txt"]
groupD=["D/standing-d.txt","D/standing_arm-d.txt","D/walking-d.txt","D/running-d.txt"]

aX, aY = load_data_from_filelist(groupA)    #Our data
bX, bY = load_data_from_filelist(groupB)
cX, cY = load_data_from_filelist(groupC)
dX, dY = load_data_from_filelist(groupD)
#get_score(RandomForestClassifier(), aX, aY, bX, bY, cX, cY, dX, dY)
cv_scores=cross_validation.cross_val_score(RandomForestClassifier(), aX, aY)
print "Cross validation Accuracy %.5f" % (np.mean(cv_scores))
print "Group B",
predict(get_selected_clf(aX, aY), bX, bY )    #get graph and score on other people's data
print "Group C",
predict(get_selected_clf(aX, aY), cX, cY )    #get graph and score on other people's data
print "Group D",
predict(get_selected_clf(aX, aY), dX, dY )    #get graph and score on other people's data