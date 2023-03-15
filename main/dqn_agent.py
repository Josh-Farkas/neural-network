import network.network as network
import numpy as np
from collections import deque
import random
from network.layers import *


class DQNAgent:
    def __init__(self, n_state, n_actions, 
                hidden_sizes = [],
                learning_rate = 0.00025,
                epsilon_decay = 0.999, 
                epsilon_min = 0.01, 
                discount_factor = 0.9,
                memory_size = 100000):

        self.n_actions = n_actions


        # initialize main network
        self.main_net = network.Network(lr=learning_rate)
        # set up main network layers
        sizes = [*hidden_sizes, n_actions]
        self.main_net.add(Dense(sizes[0], input_size=(n_state,), activation_func='relu'))
        for size in sizes[1:]:
            self.main_net.add(Dense(size, activation_func='relu'))

        # copy main network to target network to initialize
        self.target_net = network.Network(layers=self.main_net.layers, lr=self.main_net.lr)

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
            target_output[action] = target_value
            self.main_net.fit([(state, target_output)], 1, 1)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target(self):
        self.target_net.set_layers(self.main_net.get_layers())
