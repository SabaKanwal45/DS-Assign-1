import numpy as np
import math

def compute_MAE(actual_values,predicted_values):
    diff = np.array(actual_values) - np.array(predicted_values)
    sum = 0
    for i in range(len(diff)):
        sum+=abs(diff[i])
    return sum/len(diff)

def compute_MSE(actual_values,predicted_values):
    diff = np.array(actual_values) - np.array(predicted_values)
    sum = 0
    for i in range(len(diff)):
        sum+=diff[i]**2
    return sum/len(diff)

def compute_RMSE(actual_values,predicted_values):
    mse = compute_MSE(actual_values,predicted_values)
    return math.sqrt(mse)