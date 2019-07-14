from featureNormaliz import normalize_features
from loadData import loadData
from common import add_bias_term_in_data
from common import compute_cost
from common import one_training_iteration
from common import predict_instance
import matplotlib.pyplot as plt
import csv
from sklearn.model_selection import train_test_split
from computeErrors import compute_MAE
from computeErrors import compute_MSE
from computeErrors import compute_RMSE


def plot_cost_function(x_data,y_data):
    plt.xlabel("Number of iterations")
    plt.ylabel("Cost function")
    line = plt.plot(x_data,y_data)
    plt.show()


def initialize_tetas(size):
    tetas = []
    for i in range(size):
        #random.random()
        tetas.append(0.1)
    return tetas


def gradient_descent_linear_regression(train_data,learning_rate):
    # Add bias term in training data
    train_data,mean,std_dev = normalize_features(train_data)
    print(train_data)
    train_data_with_bias = add_bias_term_in_data(train_data)
    predicted_y = []
    if(len(train_data_with_bias)):
        tetas = initialize_tetas(len(train_data_with_bias[0])-1)
        tetas,change_in_tetas = one_training_iteration(train_data_with_bias,tetas,learning_rate,len(train_data_with_bias))
        cost_arr = []
        iteration = []
        for i in range(50):
            tetas,change_in_tetas = one_training_iteration(train_data_with_bias,tetas,learning_rate,len(train_data_with_bias))
            print("Iteration %d",i+1)
            iteration.append(i+1)
            cost = compute_cost(train_data_with_bias,tetas)
            cost_arr.append(cost)
            print("Cost: %(key1)s"%{'key1':cost})
        for i in range(len(train_data_with_bias)):
            predicted_y.append(predict_instance(train_data_with_bias[i],tetas))
            #print("predicted value %(key1)s Actual value %(key2)s"%{'key1':predict_instance(train_data_with_bias[i],tetas),'key2':train_data_with_bias[i][len(train_data_with_bias[0])-1]})
        plot_cost_function(iteration,cost_arr)
    else:
        print("No training Data")
    return tetas,mean,std_dev,predicted_y


def normalize_test_data(test_data,mean,std_dev):
    for i in range(len(test_data)):
        for att in range(len(test_data[0])-1):
            test_data[i][att] = (test_data[i][att] - mean[att])/std_dev[att]
    return test_data

def predict_test_data(test_data,tetas):
    test_data_with_bias = add_bias_term_in_data(test_data)
    predicted_y = []
    for i in range(len(test_data_with_bias)):
        predicted_y.append(predict_instance(test_data_with_bias[i],tetas))
    return predicted_y

def evaluateModels(data):
    train_data, test_data = train_test_split(data, test_size=0.10)
    tetas,mean,std_dev,predicted_y = gradient_descent_linear_regression(train_data,0.1)
    print("Train split")
    actual_y = [row[-1] for row in train_data]
    print("MAE %(key1)s"%{'key1':compute_MAE(actual_y,predicted_y)})
    print("MSE %(key1)s"%{'key1':compute_MSE(actual_y,predicted_y)})
    print("RMSE %(key1)s"%{'key1':compute_RMSE(actual_y,predicted_y)})
    print("Test Split")
    test_data = normalize_test_data(test_data,mean,std_dev)
    predicted_y = predict_test_data(test_data,tetas)
    actual_y = [row[-1] for row in test_data]
    print("MAE %(key1)s"%{'key1':compute_MAE(actual_y,predicted_y)})
    print("MSE %(key1)s"%{'key1':compute_MSE(actual_y,predicted_y)})
    print("RMSE %(key1)s"%{'key1':compute_RMSE(actual_y,predicted_y)})


def main():
    #ex1data1.
    train_data = loadData('data/ex1data2.txt')
    #evaluateModels(train_data)
    tetas,mean,std_dev,predicted_y = gradient_descent_linear_regression(train_data,0.5)
    test_data = loadData('data/testdata.txt')
    test_data = normalize_test_data(test_data,mean,std_dev)
    test_data = add_bias_term_in_data(test_data)
    print("Start predicting new instances")
    for i in range(len(test_data)):
        predicted_value = predict_instance(test_data[i],tetas)
        print("value of instance %(key1)s is %(key2)s"%{'key1':i+1,'key2':predicted_value})


main()