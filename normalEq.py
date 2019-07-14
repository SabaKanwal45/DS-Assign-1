from loadData import loadData
from common import add_bias_term_in_data
import numpy as np
from numpy.linalg import inv
from common import predict_instance

from sklearn.model_selection import train_test_split
from computeErrors import compute_MAE
from computeErrors import compute_MSE
from computeErrors import compute_RMSE

def training(train_data):
    temp_X = [row[:-1] for row in train_data]
    Y = np.array([row[-1] for row in train_data])
    X = add_bias_term_in_data(temp_X)
    X_t = X.transpose()
    X_mul_X_t = X_t.dot(X)
    tetas = inv(X_mul_X_t).dot(X_t).dot(Y)
    """print(X)
    print(Y)
    print(tetas)"""
    print(predict_instance([1, 1650, 3], tetas))
    return tetas

def predict_test_data(test_data,tetas):
    test_data_with_bias = add_bias_term_in_data(test_data)
    predicted_y = []
    for i in range(len(test_data_with_bias)):
        predicted_y.append(predict_instance(test_data_with_bias[i],tetas))
    return predicted_y

def evaluateModels(data):
    train_data, test_data = train_test_split(data, test_size=0.25)
    tetas = training(train_data)
    predicted_y = predict_test_data(train_data,tetas)
    print("Train split")
    actual_y = [row[-1] for row in train_data]
    print("MAE %(key1)s"%{'key1':compute_MAE(actual_y,predicted_y)})
    print("MSE %(key1)s"%{'key1':compute_MSE(actual_y,predicted_y)})
    print("RMSE %(key1)s"%{'key1':compute_RMSE(actual_y,predicted_y)})
    print("Test Split")
    predicted_y = predict_test_data(test_data,tetas)
    actual_y = [row[-1] for row in test_data]
    print("MAE %(key1)s"%{'key1':compute_MAE(actual_y,predicted_y)})
    print("MSE %(key1)s"%{'key1':compute_MSE(actual_y,predicted_y)})
    print("RMSE %(key1)s"%{'key1':compute_RMSE(actual_y,predicted_y)})


def main():
    #ex1data1.
    train_data = loadData('data/ex1data2.txt')
    #evaluateModels(train_data)
    tetas = training(train_data)
    predicted_y = predict_test_data(train_data,tetas)
    actual_y = [row[-1] for row in train_data]
    print("MAE %(key1)s"%{'key1':compute_MAE(actual_y,predicted_y)})
    print("MSE %(key1)s"%{'key1':compute_MSE(actual_y,predicted_y)})
    print("RMSE %(key1)s"%{'key1':compute_RMSE(actual_y,predicted_y)})
    #plotDatain2D(train_data)
    

main()