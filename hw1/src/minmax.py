from model import *

class Operator:
    """Base operator class for Min and Max module.
    """
    def __init__(self, operands):
        """Init operator

        Args:
            operands (list): the list of operators, including Operator or Net.
        """
        self.operands = operands
    
    def forward(self, src):
        """Foward prediction for src input

        Args:
            src (list): the input list [x,y]
        """
        pass

class Min(Operator):
    """Min module"""
    def forward(self, src):
        return float(min([x.forward(src) for x in self.operands]))

class Max(Operator):
    """Max module"""
    def forward(self, src):
        return float(max([x.forward(src) for x in self.operands]))

def divide(train_data, k=2):
    """Divide the data into positive and negative merged 2D data array.

    Args:
        train_data (array): the data to be divided
        k (int): the number of split on positive/negative set

    Returns:
        divided_data (array): the 2D data array for computation.
    """
    positive = []
    negative = []
    for entry in train_data:
        if entry[2] == 1:
            positive.append(entry)
        else:
            negative.append(entry)

    positive_folds = folds(positive, k)
    negative_folds = folds(negative, k)
    divided_data = []
    for i in range(k):
        divided_data.append([])
        for j in range(k):
            divided_data[i].append(positive_folds[i] + negative_folds[j])
    return divided_data

def trainer(train_sub_data, epochs, lr=0.05, random_seed=None):
    net = Net(lr, random_seed=random_seed)
    net = train(net, train_sub_data, epochs)
    return net

def minmax(train_data, k, epochs, lr=0.05, random_seed=None):
    divided_data = divide(train_data, k)
    
    # train the units
    subnets = []
    for i in range(k):
        subnets.append([])
        for j in range(k):
            subnets[i].append(trainer(divided_data[i][j], epochs, lr, random_seed))

    # merge to min
    mins = []
    for i in range(k):
        mins.append(Min(subnets[i]))

    # merge to max
    return Max(mins), subnets, mins

if __name__ == "__main__":
    train_data = read_data("../two_spiral_train_data.txt")
    minmax_net, subnets, mins = minmax(train_data, 3, 100, 0.05)
    test_data = read_data("../two_spiral_test_data.txt")
    print(test(minmax_net, test_data))