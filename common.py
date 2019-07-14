import numpy as np
def add_bias_term_in_data(train_data):
    x = np.array(train_data)
    y = []
    for i in range(len(train_data)):
        y.append([1])
    train_data_with_bias = np.append(y, x, axis=1)
    return train_data_with_bias


def predict_instance(instance, tetas):
    val = 0
    for index in range(len(tetas)):
        val += instance[index]*tetas[index]
    return val

def compute_cost(train_data,tetas):
    cost = 0
    for index in range(len(train_data)):
        cost += (predict_instance(train_data[index], tetas) - train_data[index][len(train_data[0])-1])**2
    return (1/(2*len(train_data))) * cost


def one_training_iteration(train_data,tetas,learning_rate,total_examples):
    temp_tetas = []
    for index in range(len(tetas)):
        diffrential_part = 0
        for ex in range(len(train_data)):
            diffrential_part += (predict_instance(train_data[ex], tetas) - train_data[ex][len(train_data[0])-1]) * train_data[ex][index]
        new_teta = tetas[index] - learning_rate * (1/total_examples) * diffrential_part
        temp_tetas.append(new_teta)
    change_in_tetas = []#abs(new_teta) - abs(temp_tetas)
    return temp_tetas,change_in_tetas

