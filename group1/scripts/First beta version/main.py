import numpy as np
import matplotlib as plt
from training import *
import plotter

def import_data (sources):
    '''
    Loads data for source and returns data read
    '''
    data_all=[]
    labels_all=[]
    for i,name in enumerate(sources):
        #Reads the "name" file
        print name
        data_t=np.genfromtxt(name,delimiter=',',skip_header=2)
    	data_all.append(data_t[:,2:5])
    return data_all

def pre_processing(raw_data, other_parameters=None):
    '''
    Pre-processes data with the selected parameters and returns them
    '''
    pass

def extract_info(pre_processed_data, other_parameters=None):
    '''
    Computes something to the data to extract what we want (in this example the norm)
    '''
    data_all=[]
    for activity_data in pre_processed_data:
        x=activity_data[:,0]
        y=activity_data[:,1]
        z=activity_data[:,2]
        norm = np.sqrt(x**2+y**2+z**2)
        x.shape = [x.shape[0], 1]
        y.shape = [x.shape[0], 1]
        z.shape = [x.shape[0], 1]
        norm.shape = [norm.shape[0], 1]
        data_all.append(np.concatenate((x, y, z, norm), axis=1))
    return data_all

def compute_windows(data_length, wlen=100, wstep=50):
    '''
    Returns the list of window starts.
    Gets lenght of signal, window length, window step
    '''
    return range(0, data_length-wlen, wstep)

def extract_features(data, wlen, wstep):
    '''
    Computes all the features
    '''
    features_all=[]
    for activity in data:
        windows_list = compute_windows(activity.shape[0], wlen, wstep)
        features_activity=[]
        for wstart in windows_list:
            wend=wstart+wstep
            data_portion=activity[wstart : wend, :]
            features_portion = get_features_row(data_portion)
            features_activity.append(features_portion)   #Add new features to features_all
        features_all.append(np.array(features_activity))
    return features_all

def get_features_row(data_portion, mean=True, std=True, min=False, max=False, ran=True):
    '''
    Computes the feature for the given data_portion
    '''
    features=[]
    for idx in range(data_portion.shape[1]):
        axis=data_portion[:,idx]
        if mean :
            features.append(np.mean(axis))
        if std :
            features.append(np.std(axis))
        if min :
            features.append(np.min(axis))
        if max :
            features.append(np.max(axis))
        if ran :
            features.append(np.max(axis)-np.min(axis))
    return np.array(features)

def get_XY(data):
    #Make a Y array for our machine learning algorithm
    data_types=[]
    for i, array in enumerate(data):
        length=array.shape[0]	#number of rows
        #Add to data_types an array containing the ID of the activity (i) for the length so that it can be used as Y
        data_types.append(np.ones(length)*i)

    Y=np.concatenate(data_types)
    X=np.concatenate(data)
    return (X, Y)

def load_data_from_filelist(filenames):
    raw_data = import_data(filenames)
    #pre_processed_data = pre_processing(raw_data)
    #useful_data = extract_info(pre_processed_data)
    useful_data =extract_info(raw_data)
    WINLEN=100
    WINSTEP=50
    data_ready = extract_features(useful_data, WINLEN, WINSTEP)
    return get_XY(data_ready)

groupA= ["A/still.txt", "A/arm.txt", "A/walk.txt", "A/run.txt"]
groupC=["C/Ema_STILL.txt", "C/Ema_STILL_ARM.txt", "C/Ema_WALK.txt", "C/Ema_RUN.txt"]
groupD=["D/standing-d.txt","D/standing_arm-d.txt","D/walking-d.txt","D/running-d.txt"]

ourX, ourY = load_data_from_filelist(groupA)
othX, othY = load_data_from_filelist(groupD)
predict_other(ourX, ourY, othX, othY)