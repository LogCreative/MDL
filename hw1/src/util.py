import numpy as np

def read_data(filename):
    """
    Reads data from the file and return an array of data
    formatting:
    [x y label]
    """
    data = []
    with open(filename, "r") as f:
        for line in f.readlines():
            x, y, label = line.split()
            x = float(x)
            y = float(y)
            label = int(label)
            data.append([x, y, label])
    return data


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))


def mse(pred, target):
    return np.mean((pred - target) ** 2)
