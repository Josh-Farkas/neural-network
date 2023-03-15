import numpy as np
import random
import mnist_loader

class Layer:
    def __init__(self):
        pass

    def feedforward(input):
        pass

    def backprop(x, y):
        pass


class Dense(Layer):
    def __init__(self, size, prev_layer_size, activation_func):
        self.activation_func = activation_func
        # self.biases = np.randn((size, 1))
        self.biases = np.zeros((size, 1))

        self.weights = np.randn((prev_layer_size, size))


    def feedforward(self, activation, bias, weight):
        """
        Calculates the activation of this layer when given the activation of the previous layer
        and the weights between the previous layer and this layer
        """
        
        if activation.ndim < 2:
            activation = np.atleast_2d(activation).transpose()

        z = np.dot(weight.transpose(), activation) + bias
        activation = self.activation_func(z)

        return activation

    def backprop(x, y):
        pass

# class Convolutional(Layer):
#      def __init__(self, k_size):
#         self.kernel = np.randn(k_size)
#         self.bias = np.randn()

#     def feedforward(self, activation):
#         pass

#     def backprop(self, output, correct):
#         for 



# class Pooling(Layer):
#     def __init__():
#         pass

#     def feedforward(input):
#         pass

#     def backprop(x, y):
#         pass


class Network:
    def __init__(self, sizes, learning_rate = 0.01, activation_function="ReLU"):

        self.num_layers = len(sizes)

        # List of vectors of correct size for every layer but the first
        self.biases = [np.random.randn(layer_size, 1) for layer_size in sizes[1:]]
        # self.biases = [np.zeros((layer_size, 1)) for layer_size in sizes[1:]]

        """
        Weights need to be a matrix where each row is an input from one node and each column is an output to one node

        Loop through all sizes 2 adjacent layers at a time
        the size of the one on the left is the width of the matrix
        the size of the one on the right is the height of the matrix
        """
        self.weights = [np.random.randn(right_layer_size, left_layer_size)
                        for right_layer_size, left_layer_size in zip(sizes[:-1], sizes[1:])]

        # self.weights = [np.zeros((right_layer_size, left_layer_size))
        #                 for right_layer_size, left_layer_size in zip(sizes[:-1], sizes[1:])]

        self.learning_rate = learning_rate
        # self.f_name = activation_function
        
        self.activation_func, self.activation_func_prime = f_list[activation_function]


    def feedforward(self, activation):
        """
        Takes in the input layer activations for the network and 
        returns the output layer activations
        """
        if activation.ndim < 2:
            activation = np.atleast_2d(activation).transpose()
        for bias, weight in zip(self.biases, self.weights):
            z = np.dot(weight.transpose(), activation) + bias
            activation = self.activation_func(z)
        return activation
            


    def SGD(self, training_data, epochs, mini_batch_size, test_data = None):
        """
        For each epoch, divide all of the training data into mini batches
        and for each mini batch apply backpropogation to each input and
        update the network.
        """
        if test_data: 
            test_data = list(test_data)
            n_test = len(test_data)

        training_data = list(training_data)

        for epoch in range(epochs):
            # randomize order of training data so you get different mini batches every epoch
            random.shuffle(training_data)

            mini_batches = [training_data[idx:idx+mini_batch_size]
                            for idx in range(0, len(training_data), mini_batch_size)]
            
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, self.learning_rate)
            
            # if you have test data return the results of this network
            if test_data:
                num_correct = 0
                # get a list of all outputs with their corresponding correct output
                outputs = [(np.argmax(self.feedforward(data)), correct)
                           for (data, correct) in test_data]
                
                num_correct = sum(int(x == y) for (x, y) in outputs)
                
                print(f"Epoch {epoch}: {num_correct}/{n_test} identified correctly")


        
    def update_mini_batch(self, mini_batch, learning_rate):
        """
        for each mini batch:
            - loop over every input in the batch
            - apply backpropogation to it and find the delta of the weights and biases
            - take the average deltas across all inputs
            - update network
        """
        # create empty matrices, without this the first addition in the sum won't work
        sum_nabla_b = [np.zeros(b.shape) for b in self.biases]
        sum_nabla_w = [np.zeros(w.shape) for w in self.weights]

        for input, correct_output in mini_batch:
            nabla_b, nabla_w = self.backprop(input, correct_output)
            # print(f"NABLA W: {nabla_w}\n\n\nSUM: {sum_nabla_w}")
            sum_nabla_b = [current + sum for current, sum in zip(nabla_b, sum_nabla_b)]
            sum_nabla_w = [current + sum for current, sum in zip(nabla_w, sum_nabla_w)]
        
        # update weights and biases based on average nabla_b and nabla_w across all inputs
        # more efficient to do b * learning_rate/len(mini_batch) so you don't need to divide
        # every element in the matrix, but this is more readable which is what I'm going for
        self.biases = [bias - learning_rate * nabla_b/len(mini_batch) 
                       for bias, nabla_b in zip(self.biases, sum_nabla_b)]
        self.weights = [weight - learning_rate * nabla_w/len(mini_batch)
                        for weight, nabla_w in zip(self.weights, sum_nabla_w)]


    def backprop(self, input_layer, correct_output):
        if input_layer.ndim < 2:
            input_layer = np.atleast_2d(input_layer).transpose()

        nabla_b = [np.zeros(layer.shape) for layer in self.biases]
        nabla_w = [np.zeros(layer.shape) for layer in self.weights]

        zs = []
        activation = input_layer
        activations = [activation] # init with the input layer already in it

        """
        feed forward through the network and store all zs and activations for use later
        this is the same math as the feedforward method
        """
        for bias, weight in zip(self.biases, self.weights):
            z = np.dot(weight.transpose(), activation) + bias
            activation = self.activation_func(z)
            zs.append(z)
            activations.append(activation)
        
        """
        Backwards pass through the network. Error of the final layer is equal to 
        Cost Derivative * Sigmoid Prime (z of the final layer). Each layer
        after that is the weight of the next layer * the error of the next layer
        * sigmoid prime of the zs of this layer
        """
        delta = cost_derivative(activations[-1], correct_output) * self.activation_func_prime(zs[-1])
        # print(activations[-1], correct_output)

        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose()).transpose()
        for l in range(2, self.num_layers):
            delta = np.dot(self.weights[-l + 1], delta) * self.activation_func_prime(zs[-l])
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose()).transpose()
        
        return (nabla_b, nabla_w)
    

    def get_weights(self):
        return self.weights
    
    def get_biases(self):
        return self.biases

    def set_weights(self, weights):
        self.weights = [weight for weight in weights]

    def set_biases(self, biases):
        self.biases = [bias for bias in biases]



def cost_derivative(output, correct_output):
    return output - correct_output


# the sigmoid function
def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

# derivative of the sigmoid function
def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))

def ReLU(z):
    # return np.maximum(np.zeros(z.shape), z)
    return z * (z > 0)

def ReLU_prime(z):
    # return np.where(z > 0, 1, 0)
    return 1 * (z > 0)

f_list = {
           "sigmoid":(sigmoid, sigmoid_prime),
           "ReLU":(ReLU, ReLU_prime) 
        }


def correlate(img, kernel, padding=0, strides=1):
    """
    Takes a filter matrix and takes the dot product with the input image as
    it moves across the whole image, producing a convolved output

    https://medium.com/analytics-vidhya/2d-convolution-using-python-numpy-43442ff5f381
    """
    img_x = img.shape[1]
    img_y = img.shape[0]
    kernel_x = kernel.shape[1]
    kernel_y = kernel.shape[0]
    output_x = int((img_x + 2*padding - kernel_x) / strides + 1)
    output_y = int((img_y + 2*padding - kernel_y) / strides + 1)

    output = np.zeros((output_y, output_x), dtype=float)

    if padding != 0:
        padded_img = np.zeros((img_y + padding*2, img_x + padding*2))
        padded_img[padding:-padding, padding:-padding] = img
    else:
        padded_img = img

    for x in range(output_x):
        for y in range(output_y):
            output[y, x] = (padded_img[y:y+kernel_y, x:x+kernel_x] * kernel).sum()
    
    return output

def convolve(img, kernel, padding=0, strides=1):
    return correlate(img, np.fliplr(np.flipud(kernel)), padding=padding, strides=strides)

def main():
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    net = Network([784, 30, 10], 3.0, activation_function="sigmoid")
    net.SGD(training_data, 30, 10, test_data)

    # a = np.array([
    #     [2, 2, 1, 4, 2],
    #     [2, 4, 1, 9, 8],
    #     [1, 1, 1, 2, 7],
    #     [1, 9, 5, 1, 0],
    #     [0, 3, 2, 5, 1]
    # ])

    # k = np.array([
    #     [2, 2, 1],
    #     [2, 4, 1],
    #     [1, 1, 1]
    # ])

    # c = correlate(a, k, padding=1)
    # print(c)



if __name__ == '__main__':
    main()