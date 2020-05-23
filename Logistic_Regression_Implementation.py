import numpy as np
from sklearn.datasets import fetch_mldata

# % matplotlib inline

import matplotlib
import matplotlib.pyplot as plt

mnist = fetch_mldata('MNIST original')
mnist

X, y = mnist["data"], mnist["target"]

X.shape
y.shape

# To know how many digits we have we can run this simple code

total = 0
# for i in range(10):
#     print("digit", i, "appear", np.count_nonzero(y == i), "times")


def plot_digit(some_digit):
    some_digit_image = some_digit.reshape(28, 28)

    plt.imshow(some_digit_image, cmap=matplotlib.cm.binary, interpolation="nearest")
    plt.axis("off")
    plt.show()


# plot_digit(X[36003])
y[36003]

# Let's first reduce our dataset only to 1 and 2 digits.

X_12 = X[np.any([y == 1, y == 2], axis=0)]
y_12 = y[np.any([y == 1, y == 2], axis=0)]

# plot_digit(X_12[8000])
# print(y_12[8000])
#
# plot_digit(X_12[9345])
# print(y_12[9345])
#
# plot_digit(X_12[877])
# print(y_12[877])
#
# plot_digit(X_12[144])
# print(y_12[144])
#
# print(X_12.shape)
# print(y_12.shape)
#
# print("number of 1:", np.count_nonzero(y_12 == 1))
# print("number of 2:", np.count_nonzero(y_12 == 2))

shuffle_index = np.random.permutation(X_12.shape[0])
X_12_shuffled, y_12_shuffled = X_12[shuffle_index], y_12[shuffle_index]

train_proportion = 0.8
train_test_cut = int(len(X_12) * train_proportion)

X_train, X_test, y_train, y_test = \
    X_12_shuffled[:train_test_cut], \
    X_12_shuffled[train_test_cut:], \
    y_12_shuffled[:train_test_cut], \
    y_12_shuffled[train_test_cut:]

# print("Shape of X_train is", X_train.shape)
# print("Shape of X_test is", X_test.shape)
# print("Shape of y_train is", y_train.shape)
# print("Shape of y_test is", y_test.shape)

np.count_nonzero(y_12 == 1) / np.count_nonzero(y_12 == 2)

print(np.count_nonzero(y_train == 1) / np.count_nonzero(y_train == 2))
print(np.count_nonzero(y_test == 1) / np.count_nonzero(y_test == 2))

X_train_normalised = X_train / 255.0
X_test_normalised = X_test / 255.0

X_train_tr = X_train_normalised.transpose()
y_train_tr = y_train.reshape(1, y_train.shape[0])
X_test_tr = X_test_normalised.transpose()
y_test_tr = y_test.reshape(1, y_test.shape[0])

# print(X_train_tr.shape)
# print(y_train_tr.shape)
# print(X_test_tr.shape)
# print(y_test_tr.shape)

dim_train = X_train_tr.shape[1]
dim_test = X_test_tr.shape[1]

print("The training dataset has dimensions equal to", dim_train)
print("The test set has dimensions equal to", dim_test)

y_train_shifted = y_train_tr - 1
y_test_shifted = y_test_tr - 1

# plot_digit(X_train_tr[:, 1005])
# print(y_train_shifted[:, 1005])
# plot_digit(X_train_tr[:, 1432])
# print(y_train_shifted[:, 1432])
# plot_digit(X_train_tr[:, 456])
# print(y_train_shifted[:, 456])
# plot_digit(X_train_tr[:, 567])
# print(y_train_shifted[:, 567])

Xtrain = X_train_tr
ytrain = y_train_shifted
Xtest = X_test_tr
ytest = y_test_shifted


def sigmoid(z):
    """
    Implement the sigmoid function

    Arguments:
    y -- a scalar (float)

    Return:
    s -- the sigmoid function evaluated on z (as in equation (1))
    """
    s = 1.0 / (1.0 + np.exp(-z))

    return s


def initialize(dim):
    """
    Initialise the weights and the bias to tensors of dimensions (dim,1) for w and
    to 1 for b (a scalar)

    Arguments:
    dim -- a scalar (float)

    Return:
    w -- a matrix of dimensions (dim,1) containing all zero
    b -- a scalar = 0
    """
    w = np.zeros((dim, 1))
    b = 0

    assert (w.shape == (dim, 1))
    assert (isinstance(b, float) or isinstance(b, int))

    return w, b


def propagate(w, b, X, Y):
    """
    Implement the cost function and its gradient for the propagation explained above

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px, 1) (our case 784,1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px, number of examples)
    Y -- true "label" vector (containing 0 if class 1, 1 if class 2) of size (1, number of examples)

    Return:
    cost -- negative log-likelihood cost for logistic regression
    dw -- gradient of the loss with respect to w, thus same shape as w
    db -- gradient of the loss with respect to b, thus same shape as b
    """

    m = X.shape[1]

    z = np.dot(w.T, X) + b
    A = sigmoid(z)
    cost = -1.0 / m * np.sum(Y * np.log(A) + (1.0 - Y) * np.log(1.0 - A))

    dw = 1.0 / m * np.dot(X, (A - Y).T)
    db = 1.0 / m * np.sum(A - Y)

    assert (dw.shape == w.shape)
    assert (db.dtype == float)

    cost = np.squeeze(cost)
    assert (cost.shape == ())

    grads = {"dw": dw,
             "db": db}

    return grads, cost


def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    """
    This function optimizes w and b by running a gradient descent algorithm

    Arguments:
    w -- weights, a numpy array of size (n_x, 1)
    b -- bias, a scalar
    X -- data of shape (n_x, m)
    Y -- true "label" vector (containing 0 if class 1, 1 if class 2), of shape (1, m)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- True to print the loss every 100 steps

    Returns:
    params -- dictionary containing the weights w and bias b
    grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
    costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.
    """
    costs = []

    for i in range(num_iterations):

        grads, cost = propagate(w, b, X, Y)

        dw = grads["dw"]
        db = grads["db"]

        w = w - learning_rate * dw
        b = b - learning_rate * db

        if i % 100 == 0:
            costs.append(cost)

        if print_cost and i % 100 == 0:
            print("Cost (iteration %i) = %f" % (i, cost))

    grads = {"dw": dw, "db": db}
    params = {"w": w, "b": b}

    return params, grads, costs


def predict(w, b, X):
    '''
    Predict whether the label is 0 or 1

    Arguments:
    w -- weights, a numpy array of size (n_x, 1)
    b -- bias, a scalar
    X -- data of size (n_x, m)

    Returns:
    Y_prediction -- a numpy array (vector) containing all predictions (0/1)
    '''

    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)

    A = sigmoid(np.dot(w.T, X) + b)

    for i in range(A.shape[1]):
        if (A[:, i] > 0.5):
            Y_prediction[:, i] = 1
        elif (A[:, i] <= 0.5):
            Y_prediction[:, i] = 0

    assert (Y_prediction.shape == (1, m))

    return Y_prediction


def model(X_train, Y_train, X_test, Y_test, num_iterations=1000, learning_rate=0.5, print_cost=False):
    w, b = initialize(X_train.shape[0])
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)

    w = parameters["w"]
    b = parameters["b"]

    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    train_accuracy = 100.0 - np.mean(np.abs(Y_prediction_train - Y_train) * 100.0)
    test_accuracy = 100.0 - np.mean(np.abs(Y_prediction_test - Y_test) * 100.0)

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train": Y_prediction_train,
         "w": w,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations}

    print("Accuarcy Test: ", test_accuracy)
    print("Accuracy Train: ", train_accuracy)

    return d

# Testmof the model

d = model(Xtrain,
          ytrain,
          Xtest,
          ytest,
          num_iterations=4000,
          learning_rate=0.05,
          print_cost=True)

# Cost function vs.number of iterations

plt.plot(d["costs"])
plt.xlim([1, 40])
plt.ylim([0, 0.12])
plt.title("Cost Function with learning rate = 0.05", fontsize=15)
plt.xlabel("No of iterations", fontsize=14)
plt.ylabel("$L(w,b)$", fontsize=17)
plt.savefig('CostFunctionPlot005.png')
plt.show()

