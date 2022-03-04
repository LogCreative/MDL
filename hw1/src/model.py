from util import *
import argparse
from tqdm import tqdm


class Net:
    """MLQP Network"""

    def __init__(self, lr=0.1, alpha=0.8, random_seed=None, hidden_num=10):
        """Init MLQP network.

        Args:
            lr (float, optional): Learning Rate. Defaults to 0.1.
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
        """Init parameters.
        """

        sqrtk = np.sqrt(1 / 2)

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

    def forward(self, x1):
        """Foward pass, return the prediction based on the given data.

        Args:
            x1 (list): the input list of data `[x,y]`
        """
        self.x1 = np.expand_dims(np.array(x1), axis=1)
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
        test_error: the mse error over test_data
    """
    return step(model, test_data, False)


def train(model, train_data, epochs):
    """Train the model by epochs.

    Args:
        model (Net): the instance of Net
        train_data (array): the training set
        epochs (int): the number of epochs
    """

    # divide the data into 3-folds.
    k = 3
    fold_size = len(train_data) // k
    fold_data = []
    for i in range(k - 1):
        fold_data.append(train_data[i * fold_size : (i + 1) * fold_size])
    fold_data.append(train_data[(k - 1) * fold_size :])

    with tqdm(total=epochs, leave=True, unit="epoch") as pbar:
        prev_val_error = np.inf
        delta = 1e-6
        epoch_range = range(model.epochs + 1, model.epochs + 1 + epochs)
        for epoch in epoch_range:
            train_error = 0.0
            val_error = 0.0
            for val in range(k):
                _train_data = []
                for i in range(k):
                    if i != val:
                        _train_data += fold_data[i]
                train_error += train_step(model, _train_data)
                _val_data = fold_data[val]
                val_error += test_step(model, _val_data)
            train_error /= k
            val_error /= k
            pbar.update(1)
            if epoch % 10 == 0:
                model.epochs = epoch
                # # Early Stop
                # if val_error - prev_val_error < -delta:
                #     prev_val_error = val_error
                # else:
                #     pbar.set_description("Early Stop %4d | T %.4f | V %.4f" % (epoch, train_error, val_error))
                #     return
                pbar.set_description(
                    "Epoch %4d | T %.4f | V %.4f" % (epoch, train_error, val_error)
                )
                model.logs.append([epoch, train_error, val_error])


def test(model, test_data):
    """Test the model

    Args:
        model (Net): the instance of Net
        test_data (arrat): the testing set

    Returns:
        test_error: the mse error over test set
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
        default=0.1,
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

    train(net, train_data, args.epochs)
    test_error = test(net, test_data)
    print("Test Error: %.4f" % test_error)
