import numpy as np
import helpers.utils as utils

class Layer:
    def __init__(self, size):
        self.size = (size,)
        self.activations = np.zeros(size)

        self.prev_layer = None


    def feedforward(self, activations):
        return activations


    def backprop(self, error):
        return error


    def on_add(self, network):
        """
        Called when this layer is added to a network,
        is passed the network it is added to
        """

        self.is_first = len(network.layers) == 1

        if not self.is_first:
            self.prev_layer = network.layers[-2]
            self.input_activations = self.prev_layer.activations

        self.lr = network.lr



class Dense(Layer):
    def __init__(self, size, activation_func, lr=None, input_size=None):
        super().__init__(size)

        self.input_size = input_size

        self.zs = np.zeros(size)
        
        # biases of this layer
        # self.biases = np.random.randn(size)
        self.biases = np.random.randn(size)

        self.activation_func, self.activation_func_prime = utils.funcs[activation_func.lower()]

        # will be initialized in on_add
        self.weights = None
        self.lr = None



    def on_add(self, network):
        super().on_add(network)
        
        prev_size = self.input_size if self.is_first else self.prev_layer.size
        self.input_activations = np.zeros(prev_size)

        # weights between previous layer and this layer
        self.weights = np.random.randn(prev_size[0], self.size[0])


    def feedforward(self, activation):
        """
        Calculates the activation of this layer when given the activation of the previous layer
        and the weights between the previous layer and this layer
        """
        self.input_activations = activation

        self.zs = np.dot(self.weights.transpose(), activation) + self.biases
        self.activations = self.activation_func(self.zs)

        return self.activations


    def backprop(self, error, batch_size):
        self.error = error
        
        nabla_b = error
        nabla_w = np.dot(np.reshape(self.input_activations, (-1, 1)), np.reshape(error, (1, -1)))

        # update weights and biases
        self.weights = self.weights - self.lr * nabla_w / batch_size
        self.biases = self.biases - self.lr * nabla_b / batch_size

        if self.is_first: return
        
        prev_layer_error = np.dot(self.weights, error) * self.activation_func_prime(self.prev_layer.zs)
        return prev_layer_error



class Convolutional(Layer):
    def __init__(self, size, kernel_size, padding=0, strides=1):
        super().__init__(size)

        self.kernel = np.random.randn(kernel_size, kernel_size)

        self.prev_layer = None
        self.input_size = None
        self.output_size = None
        self.bias = None
        self.activations = None


    def on_add(self, network):
        super().on_add(network)
        self.input_size = self.prev_layer.size
        
        output_x = (self.input_size[0] - self.kernel_size + (2 * self.padding)) / self.strides
        output_y = (self.input_size[1] - self.kernel_size + (2 * self.padding)) / self.strides
        self.output_size = (output_y, output_x)
        
        self.bias = np.random.randn(self.output_size)
        self.activations = np.zeros(self.output_size)


    def feedforward(self, activation):
        pass


    def backprop(self, output, correct):
        pass



# class Pooling(Layer):
#     def __init__():
#         pass

#     def feedforward(input):
#         pass

#     def backprop(x, y):
#         pass


class Flatten(Layer):
    def __init__(self, input_shape, size):
        super().__init__()
        self.size = size
        self.input_shape = input_shape
        activations = np.zeros(size)

    def feedforward(self, activation):
        return activation.flatten()

    def backprop(self, error):
        """
        Error will just pass through flatten layer and be reshaped because the flatten layer
        has no effect on the loss function
        """
        return error.reshape(self.input_shape)
