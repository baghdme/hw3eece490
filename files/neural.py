import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, lr=0.01):
        self.weights1 = np.random.randn(input_size, hidden_size)
        self.bias1 = np.zeros((1, hidden_size))
        self.weights2 = np.random.randn(hidden_size, output_size)
        self.bias2 = np.zeros((1, output_size))
        self.lr = lr

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, X):
        self.layer1 = self.sigmoid(np.dot(X, self.weights1) + self.bias1)
        self.output = self.sigmoid(np.dot(self.layer1, self.weights2) + self.bias2)
        return self.output

    def backward(self, X, y, output):
        output_error = y - output
        output_delta = output_error * self.sigmoid_derivative(output)

        layer1_error = output_delta.dot(self.weights2.T)
        layer1_delta = layer1_error * self.sigmoid_derivative(self.layer1)

        self.weights2 += self.layer1.T.dot(output_delta) * self.lr
        self.bias2 += np.sum(output_delta, axis=0) * self.lr
        self.weights1 += X.T.dot(layer1_delta) * self.lr
        self.bias1 += np.sum(layer1_delta, axis=0) * self.lr

    def train(self, X, y, epochs=1000):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)

    def predict(self, X):
        output = self.forward(X)
        return (output > 0.5).astype(int).flatten()