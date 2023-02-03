import numpy as np
import random
import mnist_loader

class Network:
    def __init__(self, sizes):

        self.num_layers = len(sizes)

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
            z = np.dot(weight.transpose(), activation) + bias
            activation = self.sigmoid(z)
        return activation
            


    def SGD(self, training_data, epochs, mini_batch_size, learning_rate, test_data = None):
        """
        For each epoch, divide all of the training data into mini batches
        and for each mini batch apply backpropogation to each input and
        update the network.
        """
        # print(list(test_data)[0][1])
        test_data = list(test_data)
        if test_data: n_test = len(test_data)
        training_data = list(training_data)

        for epoch in range(epochs):
            # randomize order of training data so you get different mini batches every epoch
            random.shuffle(training_data)

            mini_batches = [training_data[idx:idx+mini_batch_size]
                            for idx in range(0, len(training_data), mini_batch_size)]
            
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, learning_rate)
            
            # if you have test data return the results of this network
            if test_data:
                num_correct = 0
                # get a list of all outputs with their corresponding correct output
                outputs = [(np.argmax(self.feedforward(data)), correct)
                           for (data, correct) in test_data]
                
                # count how many outputs are equal to the correct output
                # for output, correct in outputs:
                #     if output == correct:
                #         num_correct += 1
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

        nabla_b = [np.zeros(layer.shape) for layer in self.biases]
        nabla_w = [np.zeros(layer.shape) for layer in self.weights]

        zs = []
        activation = input_layer
        activations = [activation] # init with the output layer already in it

        """
        feed forward through the network and store all zs and activations for use later
        this is the same math as the feedforward method
        """
        for bias, weight in zip(self.biases, self.weights):
            z = np.dot(weight.transpose(), activation) + bias
            activation = self.sigmoid(z)
            zs.append(z)
            activations.append(activation)
        
        """
        Backwards pass through the network. Error of the final layer is equal to 
        Cost Derivative * Sigmoid Prime (z of the final layer). Each layer
        after that is the weight of the next layer * the error of the next layer
        * sigmoid prime of the zs of this layer
        """
        delta = self.cost_derivative(activations[-1], correct_output) * self.sigmoid_prime(zs[-1])

        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose()).transpose()

        for l in range(2, self.num_layers):
            delta = np.dot(self.weights[-l + 1], delta) * self.sigmoid_prime(zs[-l])
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose()).transpose()
        
        return (nabla_b, nabla_w)



    def cost_derivative(self, output, correct_output):
        return output - correct_output

    # the sigmoid function
    def sigmoid(self, z):
        return 1.0/(1.0 + np.exp(-z))
    
    # derivative of the sigmoid function
    def sigmoid_prime(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))

def main():
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    net = Network([784, 30, 50, 10])
    net.SGD(training_data, 30, 10, 3.0, test_data)


if __name__ == '__main__':
    main()