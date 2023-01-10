from abc import abstractmethod, ABC
from typing import List
import numpy as np
from keras.utils import to_categorical
from keras.datasets import mnist
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
        self.weights = np.random.rand(self.input_size, self.output_size) - 0.5
        self.bias = np.random.rand(1, output_size) - 0.5

    def forward(self, x:np.ndarray)->np.ndarray:
        self.input = x
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def backward(self, output_error_derivative, learning_rate)->np.ndarray:
        self.learning_rate = learning_rate
        input_error = np.dot(output_error_derivative, self.weights.T)
        weights_error = np.dot(self.input.T, output_error_derivative)
        self.weights -= self.learning_rate * weights_error
        self.bias -= self.learning_rate * output_error_derivative
        return input_error

class Tanh(Layer):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x:np.ndarray)->np.ndarray:
        self.input = x
        output = np.tanh(self.input)
        return output

    def backward(self, output_error_derivative, learning_rate)->np.ndarray:
        return output_error_derivative*(1-np.tanh(self.input)**2)

class Loss:
    def __init__(self, loss_function:callable, loss_function_derivative:callable)->None:
        self.loss_function = loss_function
        self.loss_function_derivative = loss_function_derivative

    def loss(self, y_true:np.ndarray, y_pred:np.ndarray)->np.ndarray:
        """Loss function for a particular x"""
        return self.loss_function(y_true, y_pred)

    def loss_derivative(self, y_true:np.ndarray, y_pred:np.ndarray)->np.ndarray:
        """Loss function derivative for a particular x and y"""
        return self.loss_function_derivative(y_true, y_pred)

class Network:
    def __init__(self, layers:List[Layer], learning_rate:float)->None:
        self.layers = layers
        self.learning_rate = learning_rate
        self.loss = None

    def compile(self, loss:Loss)->None:
        """Define the loss function and loss function derivative"""
        self.loss = Loss

    def __call__(self, x:np.ndarray) -> np.ndarray:
        """Forward propagation of x through all layers"""
        results = []
        for data in x:
          output = data
          for layer in self.layers:
            output = layer.forward(output)
          results.append(output)
        return results

    def fit(self,
            x_train:np.ndarray,
            y_train:np.ndarray,
            epochs:int,
            learning_rate:float,
            verbose:int=0)->None:
        """Fit the network to the training data"""
        num_of_samples = len(x_train)

        for epoch in range(epochs):
            err = 0
            for i in range(num_of_samples):
                output = x_train[i]
                for layer in self.layers:
                    output = layer.forward(output)
                err += calculate_error(y_train[i], output)
                error = calculate_error_derivative(y_train[i], output)
                for layer in reversed(self.layers):
                    error = layer.backward(error, learning_rate)
            err = err / num_of_samples
            print('epoch %d/%d   error=%f' % (epoch+1, epochs, err))
    
def calculate_error(y_true, y_pred):
        return np.mean(np.power(y_true-y_pred, 2))
        
def calculate_error_derivative(y_true, y_pred):
        return 2*(y_pred-y_true)/y_true.size

def main():
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()

    # Convert y_train into one-hot format
    temp = []
    for i in range(len(Y_train)):
        temp.append(to_categorical(Y_train[i], num_classes=10))
    Y_train = np.array(temp)

    # Convert y_test into one-hot format
    temp = []
    for i in range(len(Y_test)):    
        temp.append(to_categorical(Y_test[i], num_classes=10))
    Y_test = np.array(temp)

    # Convert image into a vector
    image_vector_size = 28*28
    X_train = X_train.astype('float32')
    X_train = X_train / 255.0
    X_train = X_train.reshape(X_train.shape[0], 1, image_vector_size)
    X_test = X_test.astype('float32')
    X_test = X_test / 255.0
    X_test = X_test.reshape(X_test.shape[0], 1, image_vector_size)

    layers = []
    layers.append(FullyConnected(784, 20))
    layers.append(Tanh())
    layers.append(FullyConnected(20, 20))
    layers.append(Tanh())
    layers.append(FullyConnected(20, 10))
    layers.append(Tanh())
    net = Network(layers, learning_rate=0.01)

    print("Niewytrenowana sieć")
    print("Wartości prawdziwe:")
    print(Y_test[0:10])
    print(np.argmax(net(X_test[0:10]), axis=0))
    print("Trenowanie sieci")
    net.fit(X_train[0:1000], Y_train[0:1000], epochs=20, learning_rate=0.01)
    print("Wytrenowana sieć")
    predictions = net(X_test[0:10])
    predictions = np.argmax(predictions, axis=0)
    print(predictions)

    

    fig, axes = plt.subplots(ncols=10, sharex=False,
                         sharey=True, figsize=(20, 4))
    for i in range(10):
        axes[i].set_title(predictions[0][i])
        axes[i].imshow(X_test[i].reshape(28, 28), cmap='gray')
        axes[i].get_xaxis().set_visible(False)
        axes[i].get_yaxis().set_visible(False)
    plt.show()
      
if __name__ == "__main__":
    main()
        
# def main():
#     x_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
#     y_train = np.array([[[0]], [[1]], [[1]], [[0]]])
#     input_layer = FullyConnected(2,3)
#     first_sigmoid = Tanh()
#     output_layer = FullyConnected(3,1)
#     second_sigmoid = Tanh()
#     layers = [input_layer, first_sigmoid, output_layer, second_sigmoid]
#     network = Network(layers, 0.1)
#     print("Działanie niewytrenowanej sieci neuronowej")
#     print(network(x_train))
#     print("Trenowanie sieci")
#     network.fit(x_train, y_train, 500, 0.1)
#     print("Działanie wytrenowanej sieci")
#     print(network(x_train))
#     # digits = load_digits()
#     # Y = digits.target
#     # X = []
#     # images = digits.images
#     # for image in images:
#     #     img = image.flatten().flatten()
#     #     X.append(img)

#     # X = np.array(X)
#     # X_train, X_rem, Y_train, Y_rem = train_test_split(X, Y, test_size=0.2)
#     # X_valid, X_test, Y_valid, Y_test = train_test_split(X_rem, Y_rem, test_size=0.5)
#     # layers = []
#     # layers.append(FullyConnected(64, 10))
#     # layers.append(Tanh())
#     # layers.append(FullyConnected(10, 10))
#     # layers.append(Tanh())
#     # # layers.append(FullyConnected(24, 10))
#     # # layers.append(Tanh())
#     # net = Network(layers, learning_rate=0.01)
#     # #net.compile(Loss)
#     # net.fit(X_train, Y_train, epochs=1000, learning_rate=0.01)
#     # out = net.__call__(X_train)
#     # print(out)
#     # # plt.gray()
#     # # plt.matshow(digits.images[0])
#     # # plt.show()
if __name__ == "__main__":
    main()