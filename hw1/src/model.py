from util import *
import argparse
import copy
from tqdm import tqdm


class Net:
    """MLQP Network"""

    def __init__(self, lr=0.05, alpha=0.8, random_seed=None, hidden_num=10):
        """Init MLQP network.

        Args:
            lr (float, optional): Learning Rate. Defaults to 0.05.
            alpha (float, optional): Momentum Constant. Defaults to 0.8.
            random_seed (int, optional): Random seed for numpy. Defaults to None.
            hidden_num (int, optional): the number of neurons in the hidden layer. Defaults to 10.
        """
        self.lr = lr
        self.alpha = alpha
        self.hidden_num = hidden_num

        if random_seed is not None:
            np.random.seed(random_seed)

        self.epochs = 0
        self.logs = []
        self.param_init()

    def param_init(self):
        """Init parameters."""

        sqrtk = np.sqrt(1 / 4)

        # x, y -> hidden
        self.u2 = np.random.uniform(-sqrtk, sqrtk, (self.hidden_num, 2))
        self.v2 = np.random.uniform(-sqrtk, sqrtk, (self.hidden_num, 2))
        self.b2 = np.random.uniform(-sqrtk, sqrtk, (self.hidden_num, 1))

        self.delta_u2 = np.zeros(self.u2.shape)
        self.delta_v2 = np.zeros(self.v2.shape)
        self.delta_b2 = np.zeros(self.b2.shape)

        # hidden -> output
        self.u3 = np.random.uniform(-sqrtk, sqrtk, (1, self.hidden_num))
        self.v3 = np.random.uniform(-sqrtk, sqrtk, (1, self.hidden_num))
        self.b3 = np.random.uniform(-sqrtk, sqrtk, (1, 1))

        self.delta_u3 = np.zeros(self.u3.shape)
        self.delta_v3 = np.zeros(self.v3.shape)
        self.delta_b3 = np.zeros(self.b3.shape)

    def forward(self, src):
        """Foward pass, return the prediction based on the given data.

        Args:
            src (list): the input list of data `[x,y]`
        """
        self.x1 = np.expand_dims(np.array(src), axis=1)
        self.y1 = self.x1 * self.x1
        self.n2 = np.matmul(self.u2, self.y1) + np.matmul(self.v2, self.x1) + self.b2
        self.x2 = sigmoid(self.n2)
        self.y2 = self.x2 * self.x2
        self.n3 = np.matmul(self.u3, self.y2) + np.matmul(self.v3, self.x2) + self.b3
        self.x3 = sigmoid(self.n3)
        return self.x3

    def backward(self, pred, target):
        """Backward pass, update the parameters.
        NOTE: should run forward pass first before calling this function.

        Args:
            pred (float): prediction based on foward pass
            target (float): the target label
        """
        self.delta3 = (target - pred) * sigmoid_prime(self.n3)
        self.delta_u3 = self.alpha * self.delta_u3 + self.lr * np.matmul(
            self.delta3, self.y2.T
        )
        self.delta_v3 = self.alpha * self.delta_v3 + self.lr * np.matmul(
            self.delta3, self.x2.T
        )
        self.delta_b3 = self.alpha * self.delta_b3 + self.lr * self.delta3
        self.u3 = self.u3 + self.delta_u3
        self.v3 = self.v3 + self.delta_v3
        self.b3 = self.b3 + self.delta_b3

        self.delta2 = (
            np.matmul(self.u3.T, self.delta3) * 2 * self.x2
            + np.matmul(self.v3.T, self.delta3)
        ) * sigmoid_prime(self.n2)
        self.delta_u2 = self.alpha * self.delta_u2 + self.lr * np.matmul(
            self.delta2, self.y1.T
        )
        self.delta_v2 = self.alpha * self.delta_v2 + self.lr * np.matmul(
            self.delta2, self.x1.T
        )
        self.delta_b2 = self.alpha * self.delta_b2 + self.lr * self.delta2
        self.u2 = self.u2 + self.delta_u2
        self.v2 = self.v2 + self.delta_v2
        self.b2 = self.b2 + self.delta_b2


def step(model, data, with_grad=True):
    """Common step for data on training or testing.

    Args:
        model (Net): the instance of Net
        data (array): data for training or testing
        with_grad (bool, optional): If it needs backward process. Defaults to True.

    Returns:
        loss: the mse loss of this batch of data
    """
    pred = np.zeros((len(data)))
    target = np.zeros((len(data)))
    for i, entry in enumerate(data):
        entry_input = entry[0:2]
        entry_target = entry[2]
        entry_pred = model.forward(entry_input)
        if with_grad:
            model.backward(entry_pred, entry_target)
        pred[i] = entry_pred
        target[i] = entry_target
    return mse(pred, target)


def train_step(model, train_data):
    """Train the model for one step.

    Args:
        model (Net): the instance of Net
        train_data (array): the training data
    """
    return step(model, train_data, True)


def test_step(model, test_data):
    """Test the model for test_data

    Args:
        model (Net): the instance of Net
        test_data (array): the testing data

    Returns:
        array: the mse error over test_data
    """
    return step(model, test_data, False)


def folds(data, k):
    """divide data sequencially into k portions

    Args:
        data (array): the data to be divided
        k (int): the number of portions

    Returns:
        array: the divided data
    """

    fold_size = len(data) // k
    fold_data = []
    for i in range(k - 1):
        fold_data.append(data[i * fold_size : (i + 1) * fold_size])
    fold_data.append(data[(k - 1) * fold_size :])
    return fold_data


def split(train_data, k):
    """split train_data into k folds.

    Args:
        train_data (array): the training data
        k (int): fold number

    Returns:
        array: the splited data formatted [train_data, val_data] array.
    """

    fold_data = folds(train_data, k)

    split_data = []
    for val in range(k):
        split_train_data = []
        for i in range(k):
            if i != val:
                split_train_data += fold_data[i]
        split_data.append([split_train_data, fold_data[val]])

    return split_data


def cross_validation(model, split_data):
    """Cross validation over split data.

    Args:
        model (Net): the instance of Net
        split_data (array): the splitted data generated from folds()

    Returns:
        float, float: the mean of training error and validation error among experiments.
    """

    k = len(split_data)
    sum_train_error = 0.0
    sum_val_error = 0.0
    for _train_data, _val_data in split_data:
        sum_train_error += train_step(model, _train_data)
        sum_val_error += test_step(model, _val_data)
    return sum_train_error / k, sum_val_error / k


def train(model, train_data, epochs, test_data=None):
    """Train the model by epochs.

    Args:
        model (Net): the instance of Net
        train_data (array): the training set
        epochs (int): the number of epochs
        test_data (array, optional): if assigned, the test error will be tracked but will not go into the training process.

    Returns:
        Net: trained model
    """

    # divide the data into 3-folds.
    split_data = split(train_data, 3)

    with tqdm(total=epochs, leave=True, unit="epoch") as pbar:
        # Parameters for early stopping.
        DELTA = 1e-4
        MAX_EPOCHS = 200
        best_model = None
        best_val_error = np.inf
        best_epoch = 0

        epoch_range = range(model.epochs + 1, model.epochs + 1 + epochs)
        for epoch in epoch_range:

            train_error, val_error = cross_validation(model, split_data)
            pbar.update(1)

            if epoch == 1 or epoch % 10 == 0:
                model.epochs = epoch
                model.logs.append(
                    [epoch, train_error, val_error]
                    + (
                        []
                        if test_data is None
                        else [test_step(copy.deepcopy(model), test_data)]
                    )
                )
                # track the test performance if test_data is defined.

                # Early Stop
                if val_error - best_val_error < -DELTA:
                    best_model = copy.deepcopy(model)
                    best_val_error = val_error
                    best_epoch = epoch
                if epoch - best_epoch >= MAX_EPOCHS:
                    pbar.set_description(
                        "Early Stop %4d | T %.4f | V %.4f"
                        % (
                            best_model.logs[-1][0],
                            best_model.logs[-1][1],
                            best_model.logs[-1][2],
                        )
                    )
                    return best_model
                    # have to return the model in this function,
                    # since deepcopy fallback could not trigger the
                    # python's modifying the parameter only through
                    # copying and assigning.

                pbar.set_description(
                    "Epoch %4d | T %.4f | V %.4f" % (epoch, train_error, val_error)
                )
        pbar.set_description(
            "Best Epoch %4d | T %.4f | V %.4f"
            % (
                best_model.logs[-1][0],
                best_model.logs[-1][1],
                best_model.logs[-1][2],
            )
        )
        return best_model


def test(model, test_data):
    """Test the model

    Args:
        model (Net): the instance of Net
        test_data (arrat): the testing set

    Returns:
        float: the mse error over test set
    """
    return test_step(model, test_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        metavar="N",
        help="number of epochs to train (default: 100)",
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
        "--lr",
        type=float,
        default=0.01,
        metavar="LR",
        help="learning rate (default: 0.1)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.8,
        metavar="A",
        help="Momentum constant (default: 0.8)",
    )
    args = parser.parse_args()

    net = Net(args.lr, args.alpha)

    train_data = read_data(args.train_file)
    test_data = read_data(args.test_file)

    net = train(net, train_data, args.epochs, test_data)
    test_error = test(net, test_data)
    print("Test Error: %.4f" % test_error)
