import numpy as np
import random

def main():
    pass




class Network2:
    def __init__(self, sizes, learning_rate = 0.01):
        self.n_layers = len(sizes)
        self.biases = [np.random.randn(size) for size in sizes[1:]]
        self.weights = [np.random.randn(size, next_size)
                        for size, next_size in zip(sizes[:-1], sizes[1:])]
        self.learning_rate = learning_rate
    
    def feedforward(self, activation):
        for bias, weight in zip(self.biases, self.weights):
            z = np.dot(weight.transpose(), activation) + bias
            activation = self.sigmoid(z)
        return activation


    def SGD(self, training_data, epochs, batch_size, test_data = None):
        # update the network on mini batches

        for epoch in epochs:
            training_data = random.shuffle(training_data)
            mini_batches = [[data for data in training_data[batch_idx:batch_idx + mini_batch_size]]
                             for batch_idx in range(len(training_data), mini_batch_size)]

            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch)

            
                


            # take avg across all batches

    def update_mini_batch(self, mini_batch):
        # take mini batch and update the network based on the average nablas
        size = len(mini_batch)
        scaler = self.learning_rate / size
        for activations, correct in mini_batch:
            nabla_b, nabla_w = self.backprop(activations, correct)
            self.biases = [bias - change * scaler for bias, change in zip(self.biases, nabla_b)]
            self.weights = [weight - change * scaler for weight, change in zip(self.weights, nabla_w)]


    def backprop(self, activation, correct):
        # feedforward to get output, store zs and activations
        activations = [activation]
        zs = []
        # backpropogate to find nabla_b and nabla_w

def sigmoid():
    return

def sigmoid_prime():
    return

if __name__ == "__main__":
    main()