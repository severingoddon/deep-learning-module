import numpy as np
import sklearn
from sklearn.base import BaseEstimator, ClassifierMixin

# Versuche die Klasse so umzuformulieren, dass sie als SciKit-Klassifizierer verwendet werden kann.


def func_sigmoid(x):
    return 1/(1 + np.exp(-x))

def func_id(x):
    return x


class neuralNetwork(BaseEstimator, ClassifierMixin):
    def __init__(self,
                 eta=0.03,
                 n_input_neurons=2,
                 n_hidden_neurons=2,
                 n_output_neurons=1,
                 weights=None,
                 n_iterations=40000,
                 random_state=42):

        self.eta = eta
        self.n_input_neurons = n_input_neurons
        self.n_hidden_neurons = n_hidden_neurons
        self.n_output_neurons = n_output_neurons
        self.weights = weights
        W_IH = []
        W_HO = []

        self.n_iterations = n_iterations
        self.random_state = random_state
        random_state = sklearn.utils.check_random_state(self.random_state)

        self.f_akt = func_sigmoid
        self.g_out = func_id

        self.errors = []

        self.network = []

        self.inputLayer = np.zeros((self.n_input_neurons + 1, 5))

        self.inputLayer[0] = 1.0

        self.network.append(self.inputLayer)

        if (weights):
            W_IH = self.weights[0]
        else:
            W_IH = 2 * random_state.random_sample(
                (self.n_hidden_neurons + 1, self.n_input_neurons + 1)) - 1

        self.network.append(W_IH)

        self.hiddenLayer = np.zeros((self.n_hidden_neurons + 1, 5))

        self.hiddenLayer[0] = 1.0

        self.network.append(self.hiddenLayer)

        if (weights):
            W_HO = self.weights[1]
        else:
            W_HO = 2 * random_state.random_sample(
                (self.n_output_neurons + 1, self.n_hidden_neurons + 1)) - 1

        self.network.append(W_HO)

        self.outputLayer = np.zeros((self.n_output_neurons + 1, 5))

        self.outputLayer[0] = 0.0

        self.network.append(self.outputLayer)
        pass

    def fit(self, X, Y=None):

        delta_w_jk = []
        delta_w_ik = []

        self.errors = []

        for iteration in range(self.n_iterations):
            error = 0.0
            for x, y in zip(X, Y):
                y_hat = self.predict(x)
                diff = y - y_hat

                error += 0.5 * np.sum(diff * diff)

                net = self.network

                net[4][:, 4] = net[4][:, 3] * diff
                net[2][:, 4] = net[2][:, 3] * (np.dot(net[3][:].T, net[4][:, 4]))

                delta_w_jk = self.eta * np.outer(net[4][:, 4], net[2][:, 2].T)

                delta_w_ik = self.eta * np.outer(net[2][:, 4], net[0][:, 2].T)

                net[1][:, :] += delta_w_ik
                net[3][:, :] += delta_w_jk

            self.errors.append(error)

        pass

    def predict(self, X, y=None):

        # Input Layer

        self.network[0][:, 2] = X

        # Hidden Layer

        self.network[2][1:, 0] = np.dot(self.network[1][1:, :],
                                        self.network[0][:, 2])

        self.network[2][1:, 1] = self.f_akt(self.network[2][1:, 0])

        self.network[2][1:, 2] = self.g_out(self.network[2][1:, 1])

        self.network[2][1:, 3] = self.network[2][1:, 2] * (1.0 - self.network[2][1:, 2])

        # Output Layer

        self.network[4][1:, 0] = np.dot(self.network[3][1:, :],
                                        self.network[2][:, 2])

        self.network[4][1:, 1] = self.f_akt(self.network[4][1:, 0])

        self.network[4][1:, 2] = self.g_out(self.network[4][1:, 1])

        self.network[4][1:, 3] = self.network[4][1:, 2] * (1.0 - self.network[4][1:, 2])

        return self.network[4][:, 2]
        pass

a = neuralNetwork()
X = np.array([[1.0,1.0,1.0],[1.0,0.0,1.0],[1.0,1.0,0.0],[1.0,0.0,0.0]])
Y = np.array([[0.0,0.0], [1.0,0.0],[0.0,1.0],[0.0,0.0]])
a.fit(X,Y)

for x,y in zip(X,Y):
    print('x=: {}'.format(x))
    print('{}'.format(a.predict(x[1:2])))


