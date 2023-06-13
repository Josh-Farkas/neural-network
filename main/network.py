import numpy as np
import random
import pickle

import helpers.mnist_loader as loader
import helpers.utils as utils
from layers import *

import matplotlib.pyplot as plt


class Network:
    def __init__(self, layers=[], lr=0.0001):
        for prev, layer in zip(layers[:-1], layers[1:]):
            layer.prev_layer = prev

        self.layers = []
        self.lr = lr
        self.len = 0

        for layer in layers:
            self.add(layer)
        
        self.backprop_count = 0
        self.errors = np.array([])
        

    def plot_err(self, x_axis, y_axis):
        plt.plot(x_axis, y_axis)
        plt.title('Error')
        plt.xlabel('Iteration')
        plt.ylabel('Error')
        plt.show()


    def add(self, layer):
        self.layers.append(layer)
        layer.on_add(self)

        self.len += 1
        self.weights = [layer.weights if layer is Dense 
                        else layer.kernel if layer is Convolutional 
                        else None for layer in self.layers]
        self.biases = [layer.biases if layer is Dense 
                       or layer is Convolutional 
                       else None for layer in self.layers]


    def feedforward(self, activation):
        for layer in self.layers:
            activation = layer.feedforward(activation)

        return activation


    def fit(self, training_data, epochs, mini_batch_size, test_data = None):
        """
        For each epoch, divide all of the training data into mini batches
        and for each mini batch apply backpropogation to each input and
        update the network.
        """
        # training_data = [(x.reshape(28,28), y.flatten()) for x, y in training_data]
        # training_data = training_data[:100]

        if test_data: 
            test_data = list(test_data)
            test_data = [(x.reshape(28,28), y.flatten()) for x, y in test_data]
            # test_data = [(x.flatten(), y.flatten()) for x, y in test_data]
            n_test = len(test_data)

        for epoch in range(epochs):
            # randomize order of training data so you get different mini batches every epoch
            random.shuffle(training_data)

            mini_batches = [training_data[idx:idx+mini_batch_size]
                            for idx in range(0, len(training_data), mini_batch_size)]
            
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch)
            
            # if you have test data return the results of this network
            if test_data:
                num_correct = 0
                # get a list of all outputs with their corresponding correct output
                outputs = [(np.argmax(self.feedforward(data)), correct)
                           for (data, correct) in test_data]
                
                num_correct = sum(int(x == y) for (x, y) in outputs)
                print(f"Epoch {epoch}: {num_correct}/{n_test} identified correctly")
        
        self.weights = [layer.weights for layer in self.layers]
        self.biases = [layer.biases for layer in self.layers]
        


    def update_mini_batch(self, mini_batch):
        for input, correct in mini_batch:
            self.backprop(input, correct)
            # print(self.feedforward(input))
        
        for layer in self.layers:
            layer.update(len(mini_batch))


    def backprop(self, activation, correct):
        # feedforward so all layers store their z and activation for later use
        self.feedforward(activation)

        error = utils.cost_derivative(self.layers[-1].activations, correct) * self.layers[-1].activation_func_prime(self.layers[-1].zs)
        # print(error)
        self.errors = np.append(self.errors, np.mean(error))

        self.backprop_count += 1
        if self.backprop_count % 10000 == 0:
            self.plot_err(range(self.backprop_count), self.errors)    

        for layer in reversed(self.layers):
            error = layer.backprop(error)
        

    def get_layers(self):
        return self.layers
    
    def set_layers(self, layers):
        self.layers = layers

    def get_weights(self):
        return [layer.weights for layer in self.layers]

    def set_weights(self, weights):
        for layer, weight in zip(self.layers, weights):
            layer.weights = weight

    def get_biases(self):
        return [layer.biases for layer in self.layers]
        
    def set_biases(self, biases):
        for layer, bias, in zip(self.layers, biases):
            layer.biases = bias


    def save(self, fname):
        with open(fname, 'wb+') as f:
            pickle.dump(self, f)

        # # god awful dict comprehension, alternating dict of weights and biases
        # # arrs = {f'{name}{idx}':arr[idx]
        # #         for name, arr in zip(['w', 'b'], [self.weights, self.biases]) 
        # #         for idx in range(self.len)}


    
def load(fname):
    try:
        with open(fname, 'rb') as f:
            return pickle.load(f)

    except OSError:
        print("ERROR: File not found.")
        return


def main():
    training_data, validation_data, test_data = loader.load_data_wrapper()
    net = Network(lr=.0024)
    # net.add(Convolutional(input_size=(28, 28), kernel_size=3, activation_func="relu", padding=0, strides=1))
    # net.add(Convolutional(kernel_size=3, activation_func="relu"))
    # net.add(Pooling(2))
    # net.add(Flatten())
    net.add(Dense((100,), activation_func="relu"))
    net.add(Dense((10,), activation_func="linear"))


    net.fit(training_data, 30, 10, test_data)

    # fname = 'test.npz'
    # # with open(fname, 'wb') as f:
    # #     net.save(f)

    # with open(fname, 'rb') as f:
    #     net = Network.load(fname)

    # net.fit(training_data, 3, 10, test_data)

if __name__ == '__main__':
    main()