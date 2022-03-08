from multiprocessing import Pool, RLock, freeze_support
import random
from model import *
import time
from functools import partial


class Operator:
    """Base operator class for Min and Max module."""

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
        array: the 2D divided data array for computation.
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
    """Trainer worker

    Args:
        train_sub_data (array): the input array for training.
        epochs (int): the number threshold of epochs.
        lr (float, optional): Learning rate. Defaults to 0.05.
        random_seed (int, optional): Random Seed. Defaults to None.

    Returns:
        Net: the trained network.
    """
    net = Net(lr, random_seed=random_seed)
    net = train(net, train_sub_data, epochs)
    return net


def minmax(train_data, k, epochs, lr=0.05, random_seed=None, parallel=True):
    """Train minmax network in multiprocessing.

    Args:
        train_data (array): the training data.
        k (int): the number of split
        epochs (int): the threshold of training epochs.
        lr (float, optional): Learning rate. Defaults to 0.05.
        random_seed (int, optional): Random Seed. Defaults to None.
        parallel (bool, optional): If uses parallel training. Defaults to True.

    Returns:
        Max, array[Net], array[Min], float: Target Network, subnets, min nets, elapsed training time among units if parallel or max training time if not parallel.
    """
    # shuffle the array for random split
    if random_seed is not None:
        random.seed(random_seed)
    random.shuffle(train_data)

    divided_data = divide(train_data, k)

    if parallel:
        # train in multiprocessing.
        divided_data_flatten = [divided_data[i][j] for i in range(k) for j in range(k)]
        freeze_support()
        tqdm.set_lock(RLock())
        p = Pool(initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),))
        start_time = time.time()
        subnets_list = p.map(
            partial(trainer, epochs=epochs, lr=lr, random_seed=random_seed),
            divided_data_flatten,
        )
        elapsed_time = time.time() - start_time
        subnets = []
        for i in range(k):
            subnets.append([])
            for j in range(k):
                subnets[i].append(subnets_list[i * k + j])
    else:
        # train the processes in sequence.
        elapsed_time = 0
        subnets = []
        for i in range(k):
            subnets.append([])
            for j in range(k):
                start_time = time.time()
                subnets[i].append(trainer(divided_data[i][j], epochs, lr, random_seed))
                elapsed_time = max(time.time() - start_time, elapsed_time)

    # merge to min
    mins = []
    for i in range(k):
        mins.append(Min(subnets[i]))

    # merge to max
    return Max(mins), subnets, mins, elapsed_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--k",
        type=int,
        default=3,
        metavar="K",
        help="number of split to train (default: 3)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        metavar="N",
        help="number of epochs to train (default: 100)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.05,
        metavar="LR",
        help="learning rate (default: 0.1)",
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default="../two_spiral_train_data.txt",
        metavar="TR",
        help="training data file",
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default="../two_spiral_test_data.txt",
        metavar="TE",
        help="test data file",
    )
    parser.add_argument(
        "--parallel",
        type=bool,
        default=False,
        metavar="P",
        help="if training in multiprocessing (default: False)",
    )
    args = parser.parse_args()

    train_data = read_data(args.train_file)
    minmax_net, subnets, mins, elpased_time = minmax(
        train_data, args.k, args.epochs, args.lr, None, args.parallel
    )
    test_data = read_data(args.test_file)
    test_error = test(minmax_net, test_data)
    print("Test Error: %.4f" % test_error)
