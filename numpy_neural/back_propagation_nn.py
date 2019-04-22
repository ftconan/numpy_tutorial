"""
@file: back_propagation_nn.py
@author: magician
@date: 2019/4/22
"""
import math
import random
import numpy as np

np.seterr(all='ignore')


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def dsigmoid(y):
    return y * (1.0 - y)


def tanh(x):
    return math.tanh(x)


def dtanh(y):
    return 1 - y * y


class MLP_NeuralNetwork(object):
    """
    MLP network
    """
    def __init__(self, input, hidden, output, iterations, learning_rate, momentum, rate_decay):
        """
        :param input:
        :param hidden:
        :param output:
        :param iterations:
        :param learning_rate:
        :param momentum:
        :param rate_decay:
        """
        # initialize parameters
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.rate_decay = rate_decay

        # initialize arrays
        self.input = input + 1
        self.hidden = hidden
        self.output = output

        # set up array of 1s for activations
        self.ai = [1.0] * self.input
        self.ah = [1.0] * self.hidden
        self.ao = [1.0] * self.output

        # create randomized weights
        input_range = 1.0 / self.input ** (1/2)
        output_range = 1.0 / self.input ** (1/2)
        self.wi = np.random.normal(loc=0, scale=input_range, size=(self.input, self.hidden))
        self.wo = np.random.normal(loc=0, scale=output_range, size=(self.hidden, self.output))

        # create arrays of 0 for changes
        self.ci = np.zeros((self.input, self.hidden))
        self.co = np.zeros((self.hidden, self.output))

    def feedForward(self, inputs):
        """
        feedForward
        :param inputs:
        :return:
        """
        if len(inputs) != self.input - 1:
            raise ValueError('Wrong number of inputs you silly goose')

        # input activations
        for i in range(self.input - 1):
            self.ai[i] = inputs[i]

        # hidden activations
        for j in range(self.hidden):
            sum = 0.0
            for i in range(self.input):
                sum += self.ai[i] * self.wi[i][j]
            self.ah[j] = tanh(sum)

        # output activations
        for k in range(self.output):
            sum = 0.0
            for j in range(self.hidden):
                sum += self.ah[j] * self.wo[j][k]
            self.ao[k] = sigmoid(sum)

        return self.ao[:]

    def backPropagate(self, targets):
        """
        backPropagate
        :param targets:
        :return:
        """
        if len(targets) != self.output:
            raise TypeError('Wrong number of targets you silly goose')

        # calculate error terms for output
        output_deltas = [0.0] * self.output
        for k in range(self.output):
            error = -(targets[k] - self.ao[k])
            output_deltas[k] = dsigmoid(self.ao[k]) * error

        # calculate error terms for hidden
        hidden_deltas = [0.0] * self.hidden
        for j in range(self.hidden):
            error = 0
            for k in range(self.output):
                error += output_deltas[k] * self.wo[j][k]

            hidden_deltas[j] = dtanh(self.ah[j]) * error

        # update the weights connecting hidden to output
        for j in range(self.hidden):
            for k in range(self.output):
                change = output_deltas[k] * self.ah[j]
                self.wo[j][k] -= self.learning_rate * change + self.co[j][k] * self.momentum
                self.co[j][k] = change

        # update the weights connecting input to hidden
        for i in range(self.input):
            for j in range(self.hidden):
                change = hidden_deltas[j] * self.ai[i]
                self.wi[i][j] -= self.learning_rate * change + self.ci[i][j] * self.momentum
                self.ci[i][j] = change

        # calculate error
        error = 0.0
        for k in range(len(targets)):
            error += 0.5 * (targets[k] - self.ao[k]) ** 2

        return error

    def test(self, patterns):
        """
        test
        :param patterns:
        :return:
        """
        for p in patterns:
            print(p[1], '->', self.feedForward(p[0]))

    def train(self, patterns):
        for i in range(self.iterations):
            error = 0.0
            random.shuffle(patterns)
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                self.feedForward(inputs)
                error += self.backPropagate(targets)

            with open('../data/error.txt', 'a') as errorfile:
                errorfile.write(str(error) + '\n')
                errorfile.close()

            if i % 10 == 0:
                print('error %-.5f' % error)
            self.learning_rate = self.learning_rate * (
                        self.learning_rate / (self.learning_rate + (self.learning_rate + self.rate_decay)))

    def predict(self, X):
        """
        predict
        :param X:
        :return:
        """
        predictions = []
        for p in X:
            predictions.append(self.feedForward(p))

        return predictions


def demo():
    """
    demo
    :return:
    """
    def load_data():
        data = np.loadtxt('../data/sklearn_digits.csv', delimiter=',')
        y = data[:, 0:10]

        data = data[:, 10:]
        data -= data.min()
        data /= data.max()

        out = []
        print(data.shape)

        # populate the tuple list with the data
        for i in range(data.shape[0]):
            fart = list((data[i,:].tolist(), y[i].tolist()))
            out.append(fart)

        return out

    X = load_data()
    print(X[9])
    NN = MLP_NeuralNetwork(64, 100, 10, iterations=50, learning_rate=0.5, momentum=0.5, rate_decay=0.01)
    NN.train(X)
    NN.test(X)


if __name__ == '__main__':
    demo()
