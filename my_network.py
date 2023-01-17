import numpy as np
import random

class MyNetwork:
    def __init__(self, sizes):
        
        # List of vectors of correct size for every layer but the first
        self.biases = [np.random.randn(layer_size, 1) for layer_size in sizes[1:]]

        """
        Weights need to be a matrix where each row is an input from one node and each column is an output to one node

        Loop through all sizes 2 adjacent layers at a time
        the size of the one on the left is the width of the matrix
        the size of the one on the right is the height of the matrix
        """
        self.weights = [np.random.randn(right_layer_size, left_layer_size)
                        for right_layer_size, left_layer_size in zip(sizes[:-1], sizes[1:])]


    def feedforward(self, activation):
        """
        Takes in the input layer activations for the network and 
        returns the output layer activations
        """
        for bias, weight in zip(self.biases, self.weights):
            z = np.dot(activation, weight) + bias
            activation = self.sigmoid(z)
        return activation
            




    def SGD(self, epochs, training_data, mini_batch_size, learning_rate, test_data = None):
        """
        For each epoch, divide all of the training data into mini batches
        and for each mini batch apply backpropogation to each input and
        update the network.
        """

        for epoch in range(epochs):
            # randomize order of training data so you get different mini batches every epoch
            random.shuffle(training_data)

            mini_batches = [training_data[idx:idx+mini_batch_size]
                            for idx in range(0, len(training_data), mini_batch_size)]
            
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch)

        
    def update_mini_batch(self, mini_batch, learning_rate):
        """
        for each mini batch:
            - loop over every input in the batch
            - apply backpropogation to it and find the delta of the weights and biases
            - take the average deltas across all inputs
            - update network
        """
        # create empty arrays of matrices that will hold the sum of all nudges in b and w
        sum_nabla_b = [np.zeros(b.shape) for b in self.biases]
        sum_nabla_w = [np.zeros(w.shape) for w in self.weights]

        for input, correct_output in mini_batch:
            output_layer = self.feedforward(input)
            nabla_b, nabla_w = self.backprop(output_layer, correct_output)
            sum_nabla_b = [current + sum for current, sum in zip(nabla_b, sum_nabla_b)]
            sum_nabla_w = [current + sum for current, sum in zip(nabla_w, sum_nabla_w)]
        
        # find the average of all of them
        self.biases = [b - learning_rate * nb/len(mini_batch) for b, nb in zip(self.biases, sum_nabla_b)]
        self.weights = [learning_rate * a/len(mini_batch) for a in sum_nabla_w]


    def backprop(self, output_layer, correct_output):

        nabla_b = [np.zeros(layer.shape) for layer in self.biases]
        nabla_w = [np.zeros(layer.shape) for layer in self.weights]

        zs = []
        activation = output_layer
        activations = [activation] # init with the output layer already in it

        """
        feed forward through the network and store all zs and activations for use later
        this is the same math as the feedforward method
        """
        for bias, weight in zip(self.biases, self.weights):
            z = np.dot(activation, weight)
            activation = self.sigmoid(z)
            zs.append(z)
            activations.append(activation)
        
        """
        Backwards pass through the network. Error of the final layer is equal to 
        """

    def cost_derivative(output, correct_output):
        return correct_output - output

    # the sigmoid function
    def sigmoid(self, z):
        return 1.0/(1.0 + np.exp(-z))
    
    # derivative of the sigmoid function
    def sigmoid_prime(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))