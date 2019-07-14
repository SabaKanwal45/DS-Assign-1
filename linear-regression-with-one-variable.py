import numpy as np
import random
import matplotlib.pyplot as plt
from loadData import loadData
from common import add_bias_term_in_data
from common import compute_cost
from common import one_training_iteration
from common import predict_instance
from computeErrors import compute_MAE
from computeErrors import compute_MSE
from computeErrors import compute_RMSE
from sklearn.model_selection import train_test_split

def plot_cost_function(x_data,y_data):
    plt.xlabel("Number of iterations")
    plt.ylabel("Cost function")
    line = plt.plot(x_data,y_data)
    plt.show()

def plotDatain2D(train_data,predicted_y):
    if(len(train_data)):
        if(len(train_data[0])==2):
            data = np.array(train_data)
            population = data[:,0]
            profit = data[:,1]
            scatter = plt.scatter(population,profit,marker='x',color='r')
            plt.xlabel("Population of city in 10,000s")
            plt.ylabel("Profit in $10,000s")
            if(len(predicted_y)):
                line = plt.plot(population,predicted_y)
                plt.legend([scatter,line[0]],['Training Data','Linear Regression'],scatterpoints=1,loc='lower right',ncol=1,fontsize=8)
                plt.title("Training Data with Linear regression fit")
                #txt="Training Data with Linear regression fit"
                #plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=12)
            else:
                plt.title("Scatter Plot of training data")
            plt.show()

def plot_linear_regression_with_one_variable(train_data,predicted_y):
    plotDatain2D(train_data,predicted_y)

def initialize_tetas(size):
    tetas = []
    for i in range(size):
        #random.random()
        tetas.append(0.01)
    return tetas

def gradient_descent_linear_regression(train_data,learning_rate):
    # Add bias term in training data
    train_data_with_bias = add_bias_term_in_data(train_data)
    predicted_y = []
    tetas = initialize_tetas(len(train_data_with_bias[0])-1)
    if(len(train_data_with_bias)):
        cost_arr = []
        iteration = []
        tetas,change_in_tetas = one_training_iteration(train_data_with_bias,tetas,learning_rate,len(train_data_with_bias))
        for i in range(1000):
            tetas,change_in_tetas = one_training_iteration(train_data_with_bias,tetas,learning_rate,len(train_data_with_bias))
            print("Iteration %d",(i+1))
            cost = compute_cost(train_data_with_bias,tetas)
            print("Cost: %(key1)s"%{'key1':cost})
            iteration.append(i+1)
            cost_arr.append(cost)
        for i in range(len(train_data_with_bias)):
            predicted_y.append(predict_instance(train_data_with_bias[i],tetas))
            #print("predicted value %(key1)s Actual value %(key2)s"%{'key1':predict_instance(train_data_with_bias[i],tetas),'key2':train_data_with_bias[i][len(train_data_with_bias[0])-1]})
        plot_linear_regression_with_one_variable(train_data,predicted_y)
        plot_cost_function(iteration,cost_arr)
    else:
        print("No training data")
    return predicted_y,tetas

def predict_test_data(test_data,tetas):
    test_data_with_bias = add_bias_term_in_data(test_data)
    predicted_y = []
    for i in range(len(test_data_with_bias)):
        predicted_y.append(predict_instance(test_data_with_bias[i],tetas))
    return predicted_y

def evaluateModels(data):
    train_data, test_data = train_test_split(data, test_size=0.50)
    predicted_y,tetas = gradient_descent_linear_regression(train_data,0.01)
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
    train_data = loadData('data/ex1data1.txt')
    #evaluateModels(train_data)
    plotDatain2D(train_data,[])
    predicted_y,tetas = gradient_descent_linear_regression(train_data,0.01)
    actual_y = [row[-1] for row in train_data]
    print("MAE %(key1)s"%{'key1':compute_MAE(actual_y,predicted_y)})
    print("MSE %(key1)s"%{'key1':compute_MSE(actual_y,predicted_y)})
    print("RMSE %(key1)s"%{'key1':compute_RMSE(actual_y,predicted_y)})
    


main()