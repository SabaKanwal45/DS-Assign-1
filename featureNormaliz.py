import math
import numpy as np


def compute_mean(attribute):
    sum = 0
    for i in range(len(attribute)):
        sum += attribute[i]
    return sum/len(attribute)

def compute_std_dev(attribute):
    mean = compute_mean(attribute)
    temp = []
    for i in range(len(attribute)):
        temp.append((attribute[i] - mean)**2)
    variance = compute_mean(temp)
    return math.sqrt(variance)

def normalize_features(train_data):
    mean = []
    std_dev = []
    for i in range(len(train_data[0])-1):
        data = np.array(train_data)
        attribute = data[:,i]
        mean1 = compute_mean(attribute)
        std_dev1 = compute_std_dev(attribute)
        """print(i+1)
        print(mean1)
        print(std_dev1)"""
        mean.append(mean1)
        std_dev.append(std_dev1)
    for i in range(len(train_data)):
        for att in range(len(train_data[0])-1):
            train_data[i][att] = (train_data[i][att] - mean[att])/std_dev[att]
    return train_data,mean,std_dev


