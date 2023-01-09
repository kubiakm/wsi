from abc import abstractmethod, ABC
from typing import List
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt

class Layer(ABC):
    """Basic building block of the Neural Network"""

    def __init__(self) -> None:
        self._learning_rate = 0.01

    @abstractmethod
    def forward(self, x:np.ndarray)->np.ndarray:
        """Forward propagation of x through layer"""
        raise NotImplementedError

    @abstractmethod
    def backward(self, output_error_derivative) ->np.ndarray:
        """Backward propagation of output_error_derivative through layer"""
        raise NotImplementedError

    @property
    def learning_rate(self):
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, learning_rate):
        assert learning_rate < 1, f"Given learning_rate={learning_rate} is larger than 1"
        assert learning_rate > 0, f"Given learning_rate={learning_rate} is smaller than 0"
        self._learning_rate = learning_rate

class FullyConnected(Layer):
    def __init__(self, input_size:int, output_size:int) -> None:
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.uniform(-1, 1, (self.input_size, self.output_size))
        self.bias = np.random.rand(1, output_size) - 0.5

    def forward(self, x:np.ndarray)->np.ndarray:
        self.input = x
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def backward(self, output_error_derivative)->np.ndarray:
        input_error = np.dot(output_error_derivative, self.weights.T)
        weights_error = np.dot(self.input.T, output_error_derivative)
        self.weights -= self._learning_rate * weights_error
        self.bias -= self._learning_rate * output_error_derivative
        return input_error

class Tanh(Layer):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x:np.ndarray)->np.ndarray:
        return np.tanh(x)

    def backward(self, output_error_derivative)->np.ndarray:
        return 1-np.tanh(output_error_derivative)**2

class Loss:
    def __init__(self, loss_function:callable, loss_function_derivative:callable)->None:
        self.loss_function = loss_function
        self.loss_function_derivative = loss_function_derivative

    def loss(self, y_true:np.ndarray, y_pred:np.ndarray)->np.ndarray:
        """Loss function for a particular x"""
        self.loss_function = np.mean(np.power(y_true-y_pred, 2))
        return self.loss_function

    def loss_derivative(self, y_true:np.ndarray, y_pred:np.ndarray)->np.ndarray:
        """Loss function derivative for a particular x and y"""
        self.loss_derivative = 2*(y_pred - y_true)/y_true.size
        return self.loss_function_derivative

class Network:
    def __init__(self, layers:List[Layer], learning_rate:float)->None:
        self.layers = layers
        self.learning_rate = learning_rate

    def compile(self, loss:Loss, y_true, y_pred)->None:
        """Define the loss function and loss function derivative"""
        loss = Loss
        
        self.loss = loss.loss_function(y_true, y_pred)
        self.loss_derivative = loss.loss_function_derivative(y_true, y_pred)

    def __call__(self, x:np.ndarray) -> np.ndarray:
        """Forward propagation of x through all layers"""
        samples = len(x)
        result = []
        for i in range(samples):
            out = x[i]
            for layer in self.layers:
                output = layer.forward(out)
            result.append(output)
        return result

    def fit(self,
            x_train:np.ndarray,
            y_train:np.ndarray,
            epochs:int,
            learning_rate:float,
            verbose:int=0)->None:
        """Fit the network to the training data"""
        samples = len(x_train)
        for i in range(epochs):
            err = 0
            for j in range(samples):
                x = x_train[j]
                for layer in self.layers:
                  output = layer.forward(x)
                err += self.loss(y_train[j], output)

                #backward propagation
                error = self.loss_derivative(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.backward(error, learning_rate)
            err /= samples
            print('epoch %d/%d   error=%f' % (i+1, epochs, err))

def main():
    digits = load_digits()
    Y = digits.target
    X = []
    images = digits.images
    for image in images:
        img = image.flatten().flatten()
        X.append(img)

    X = np.array(X)
    X_train, X_rem, Y_train, Y_rem = train_test_split(X, Y, test_size=0.2)
    X_valid, X_test, Y_valid, Y_test = train_test_split(X_rem, Y_rem, test_size=0.5)
    layers = []
    layers.append(FullyConnected(64, 10))
    layers.append(Tanh)
    layers.append(FullyConnected(64, 10))
    layers.append(Tanh)
    net = Network(layers, learning_rate=0.01)
    #net.compile(Loss)
    net.fit(X_train, Y_train, epochs=1000, learning_rate=0.01)
    out = net.predict(X_train)
    print(out)
    # plt.gray()
    # plt.matshow(digits.images[0])
    # plt.show()
if __name__ == "__main__":
    main()