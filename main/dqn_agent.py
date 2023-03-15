import deprecated.old_net as network
import numpy as np
from collections import deque
import random


class DQNAgent:
    def __init__(self, n_state, n_actions, hidden_sizes = [10],
                learning_rate = 0.01,
                epsilon_decay = 0.996, 
                epsilon_min = 0.01, 
                discount_factor = 0.9,
                memory_size = 100000):

        self.n_actions = n_actions

        self.main_net = network.Network([n_state, *hidden_sizes, n_actions], learning_rate, "ReLU")
        self.target_net = network.Network([n_state, *hidden_sizes, n_actions], learning_rate, "ReLU")
        self.target_net.set_biases(self.main_net.get_biases())
        self.target_net.set_weights(self.main_net.get_weights())

        self.epsilon = 1
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.discount_factor = discount_factor

        self.replay_buffer = deque(maxlen=memory_size)


    def store_memory(self, state, action, reward, next_state, terminal):
        self.replay_buffer.append((state, action, reward, next_state, terminal))
    
    def choose_action(self, state):
        # print(self.epsilon)
        n = random.random()
        if n < self.epsilon:
            action = random.randrange(0, self.n_actions)
        else:
            action = np.argmax(self.target_net.feedforward(state))

        return action
        
    def train(self, batch_size):
        # if len(self.replay_buffer) < batch_size: return
        if len(self.replay_buffer) < batch_size:
            return
        mini_batch = random.sample(self.replay_buffer, batch_size)
        for state, action, reward, next_state, terminal in mini_batch:
            if terminal:
                target_value = reward
            else:
                target_value = reward + self.discount_factor * np.amax(self.target_net.feedforward(next_state))
            
            target_output = self.main_net.feedforward(state)
            print(f'Value: {target_value} \nPredicted: {target_output[action][0]}')
            target_output[action] = target_value
            self.main_net.SGD([(state, target_output)], 1, 1)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target(self):
        self.target_net.set_biases(self.main_net.get_biases())
        self.target_net.set_weights(self.main_net.get_weights())