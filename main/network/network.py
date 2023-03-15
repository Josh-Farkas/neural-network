import numpy as np
import random
import helpers.mnist_loader as loader
import helpers.utils as utils
import layers


class Network:
    def __init__(self, layers=[], lr=0.001):
        for prev, layer in zip(layers[:-1], layers[1:]):
            layer.prev_layer = prev

        self.layers = layers
        self.lr = lr


    def add(self, layer):
        self.layers.append(layer)
        layer.on_add(self)


    def feedforward(self, activation):
        pass
        for layer in self.layers:
            activation = layer.feedforward(activation)

        return activation


    def SGD(self, training_data, epochs, mini_batch_size, test_data = None):
        """
        For each epoch, divide all of the training data into mini batches
        and for each mini batch apply backpropogation to each input and
        update the network.
        """
        training_data = [(x.flatten(), y.flatten()) for x, y in training_data]
        test_data = [(x.flatten(), y.flatten()) for x, y in test_data]

        if test_data: 
            test_data = list(test_data)
            n_test = len(test_data)

        if training_data is not list:
            training_data = list(training_data)

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


    def update_mini_batch(self, mini_batch):
        for input, correct in mini_batch:
            self.backprop(input, correct, len(mini_batch))


    def backprop(self, activation, correct, batch_size):
        # feedforward so all layers store their z and activation for later use
        self.feedforward(activation)

        error = utils.cost_derivative(self.layers[-1].activations, correct) * self.layers[-1].activation_func_prime(self.layers[-1].zs)

        # print(f'Last Layer Error: {error}')

        for layer in reversed(self.layers):
            error = layer.backprop(error, batch_size)



def main():
    training_data, validation_data, test_data = loader.load_data_wrapper()
    # net = Network([784, 30, 50, 10], 3.0, 10)
    net = Network(lr=3.)
    net.add(layers.Dense(30, activation_func="sigmoid", input_size = (784,)))
    # net.add_layer(layers.Dense(30, activation_func_name="ReLU"))
    net.add(layers.Dense(10, activation_func="sigmoid"))

    net.SGD(training_data, 30, 10, test_data)



if __name__ == '__main__':
    main()