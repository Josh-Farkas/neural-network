import numpy as np
import helpers.utils as utils

class Layer:
    """
    A single layer in the network, this class itself won't do anything,
    it should be extended.
    """
    def __init__(self, size, dtype=np.float32, lower_bound=None, upper_bound=None):
        self.size = size
        self.activations = np.zeros(size, dtype=dtype)
        self.dtype = dtype
        self.error = np.zeros(size, dtype=dtype)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        if lower_bound == None:
            self.lower_bound = np.finfo(dtype).min
        if upper_bound == None:
            self.upper_bound = np.finfo(dtype).max

        self.prev_layer = None


    def feedforward(self, activations):
        """
        Feedforward the activation coming into this layer into the next layer
        """
        return activations


    def backprop(self, error):
        """
        Propogate the error of this layer backwards through the network
        """
        return error


    def on_add(self, network):
        """
        Called when this layer is added to a network, used to initialize
        variables that require information on the network it is being added to
        such as the size of previous layers
        """

        self.is_first = len(network.layers) == 1

        if not self.is_first:
            self.prev_layer = network.layers[-2]
            self.input_activations = self.prev_layer.activations

        self.lr = network.lr
    
    def update(self, _):
        pass


class Dense(Layer):
    def __init__(self, size, activation_func, lr=None, input_size=None, dtype=np.float16):
        super().__init__(size, dtype)

        self.input_size = input_size

        self.zs = np.zeros(size, dtype=dtype)
        
        # biases of this layer
        # self.biases = np.random.randn(size)
        self.biases = np.random.randn(*size).astype(dtype)
        self.nabla_b = np.zeros(self.biases.shape, dtype=dtype)

        self.activation_func, self.activation_func_prime = utils.funcs[activation_func.lower()]

        # will be initialized in on_add
        self.weights = None
        self.lr = lr



    def on_add(self, network):
        super().on_add(network)
        
        prev_size = self.input_size if self.is_first else self.prev_layer.size
        self.input_activations = np.zeros(prev_size)

        # weights between previous layer and this layer
        self.weights = np.random.randn(*prev_size, *self.size).astype(self.dtype)
        self.nabla_w = np.zeros(self.weights.shape, dtype=self.dtype)


    def feedforward(self, activation):
        """
        Calculates the activation of this layer when given the activation of the previous layer
        and the weights between the previous layer and this layer
        """
        self.input_activations = activation

        self.zs = np.dot(self.weights.transpose(), activation) + self.biases
        self.activations = self.activation_func(self.zs)

        return self.activations


    def backprop(self, error):
        self.error = error
        
        self.nabla_b += error
        self.nabla_b = np.clip(self.nabla_b, self.lower_bound, self.upper_bound)

        self.nabla_w += np.dot(np.reshape(self.input_activations, (-1, 1)), np.reshape(error, (1, -1)))
        self.nabla_w = np.clip(self.nabla_w, self.lower_bound, self.upper_bound)
        # TODO: Fix inf error
        # issue is that numbers are getting too large for the float16...
        # research way to efficiently limit size of numbers in array to
        # avoid -inf and inf

        if self.is_first: 
            return

        prev_layer_error = np.dot(self.weights, error) * self.activation_func_prime(self.prev_layer.zs)
        return prev_layer_error

    def update(self, batch_size):
        self.biases = self.biases - self.lr * self.nabla_b / batch_size
        self.biases[self.biases < self.lower_bound] = self.lower_bound
        self.biases[self.biases > self.upper_bound] = self.upper_bound

        self.weights = self.weights - self.lr * self.nabla_w / batch_size
        self.weights[self.weights < self.lower_bound] = self.lower_bound
        self.weights[self.weights > self.upper_bound] = self.upper_bound

        # update and clip weights to be between bounds
        # self.biases = np.clip(self.biases - self.lr * self.nabla_b / batch_size, \
        #         self.lower_bound, self.upper_bound)

        # self.weights = np.clip(self.weights - self.lr * self.nabla_w / batch_size, \
        #         self.lower_bound, self.upper_bound)

        self.nabla_b = np.zeros(self.nabla_b.shape, dtype=self.dtype)
        self.nabla_w = np.zeros(self.nabla_w.shape, dtype=self.dtype)



class Convolutional(Layer):
    def __init__(self, kernel_size, activation_func, padding=0, strides=1, lr=None, input_size=None, dtype=np.float32):
        super().__init__(size=0, dtype=dtype)

        self.activation_func = utils.funcs[activation_func]

        self.kernel_size = kernel_size
        self.kernel = np.random.randn(*kernel_size)
        self.padding = padding
        self.strides = strides

        self.prev_layer = None
        self.input_size = input_size
        self.output_size = None
        self.biases = None
        self.activations = None
        self.lr = lr



    def on_add(self, network):
        super().on_add(network)
        if self.input_size == None:
            self.input_size = self.prev_layer.size
        
        output_x = int((self.input_size[0] - self.kernel_size + (2 * self.padding)) / self.strides + 1)
        output_y = int((self.input_size[1] - self.kernel_size + (2 * self.padding)) / self.strides + 1)
        self.output_size = (output_y, output_x)
        self.size = self.output_size
        
        self.biases = np.random.randn(*self.output_size)
        self.nabla_b = np.zeros(self.output_size)
        self.nabla_k = np.zeros((self.kernel_size, self.kernel_size))

        self.activations = np.zeros(self.output_size)


    def feedforward(self, activation):
        self.activations = activation
        return utils.correlate(activation, self.kernel, self.padding, self.strides)
        # return map(self.activation_func, utils.correlate(activation, self.kernel, self.padding, self.strides))


    def backprop(self, error):
        self.error = error
        # map(self.activation_func_prime, error)

        self.nabla_b += error

        # kernel error is just the cross correlated error between activations and error
        self.nabla_k += utils.correlate(self.activations, error, self.padding, self.strides)

        # get the error of the previous layer by convolving the error
        # with the kernel
        return utils.convolve(error, self.kernel, self.kernel_size-1)

    
    def update(self, batch_size):
        # update weights and biases
        self.biases = self.biases - self.lr * self.nabla_b / batch_size
        self.kernel = self.kernel - self.lr * self.nabla_k / batch_size

        self.nabla_b = np.zeros(self.nabla_b.shape, dtype=self.dtype)
        self.nabla_k = np.zeros(self.nabla_k.shape, dtype=self.dtype)
        


class Pooling(Layer):
    def __init__(self, pool_size, type="max"):
        super().__init__(size=0)
        self.pool_size = pool_size
    
    def on_add(self, network):
        super().on_add(network)
        self.size = (int((self.prev_layer.size[0] - self.pool_size) / self.pool_size + 1),
                     int((self.prev_layer.size[0] - self.pool_size) / self.pool_size + 1))
        
        # mask of which input elements were pooled for use in backprop (1 -> was max, 0 -> wasn't max)
        self.mask = np.zeros(self.prev_layer.size)

    def feedforward(self, activation):
        self.activations, self.mask = utils.maxpool(activation, self.pool_size)
        return self.activations

    def backprop(self, error):
        self.error = error
        # gradient is 0 for non-pooled values and 1 for pooled values
        return np.multiply(error, self.mask)



class Flatten(Layer):
    def __init__(self, dtype=np.float32):
        super().__init__(0, dtype)

    def on_add(self, network):
        super().on_add(network)
        self.input_shape = self.prev_layer.size
        self.activations = np.ravel(np.zeros(self.input_shape))
        self.zs = self.activations
        self.size = self.activations.shape


    def feedforward(self, activation):
        self.activations = activation.flatten()
        self.zs = activation.flatten()
        return self.activations


    def backprop(self, error):
        """
        Error will just pass through flatten layer and be reshaped because the flatten layer
        has no effect on the loss function
        """
        self.error = error
        return error.reshape(self.input_shape)
