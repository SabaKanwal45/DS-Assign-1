import csv

def loadData(filename):
    train_data = []
    with open(filename, 'r') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(len(dataset)):
            results = list(map(float, dataset[x][0:len(dataset[0])]))
            train_data.append(results)
    return train_data