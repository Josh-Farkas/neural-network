import network as network
import numpy as np
from collections import deque
import random
from layers import *
import pickle


class DQNAgent:
    def __init__(self, n_actions, 
                layers=[],
                lr = 0.00025,
                epsilon_decay = 0.996, 
                epsilon_min = 0.01, 
                discount_factor = 0.9,
                memory_size = 100000,
                replay_buffer = None,
                action_intervals = []):

        # initialize main network
        self.main_net = network.Network(lr=lr, layers=layers)

        # copy main network to target network to initialize
        self.target_net = network.Network(layers=self.main_net.layers, lr=self.main_net.lr)

        self.n_actions = n_actions
        self.epsilon = 1
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.discount_factor = discount_factor

        if replay_buffer == None:
            self.replay_buffer = deque(maxlen=memory_size)
        else:
            self.replay_buffer = replay_buffer


    def remember(self, state, action, reward, next_state, terminal):
        self.replay_buffer.append((state, action, reward, next_state, terminal))
    
    
    def act(self, state):
        n = random.random()
        if n < self.epsilon: # take random action
            return random.randrange(0, self.n_actions)
        
        # take best action
        # print(self.target_net.feedforward(state))
        # print(f'max:{np.argmax(self.target_net.feedforward(state))}')
        return np.argmax(self.target_net.feedforward(state))
    

    def act_multiple(self, state, intervals):
        # WIP DONT USE
        # continuous actions in an interval, returns array of all actions for when multiple actions can be taken at once
        n = random.random()
        if n < self.epsilon:
            return [(random.random() * (interval[1] - interval[0]) + interval[0]) for interval in intervals]
        
        return self.target_net.feedforward(state)
        

        
    def train(self, batch_size):
        if len(self.replay_buffer) < batch_size: return

        mini_batch = random.sample(self.replay_buffer, batch_size)
        for state, action, reward, next_state, terminal in mini_batch:
            if terminal:
                target_value = reward
            else:
                target_value = self.find_target_value(reward, next_state)
            
            target_output = self.main_net.feedforward(state)
            target_output[action] = target_value

            self.main_net.fit([(state, target_output)], 1, 1)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


    def find_target_value(self, reward, next_state):
        return reward + self.discount_factor * np.amax(self.target_net.feedforward(next_state))


    def update_target(self):
        self.target_net.set_layers(self.main_net.get_layers())
    
    
    def save(self, fname):
        with open(fname, 'wb+') as f:
            pickle.dump(self, f)



class DoubleDQNAgent(DQNAgent):

    # The function to find the target value of the Q function
    # is very similar to a regular DQN. The only difference is
    # that it takes Q_target(S_t+1, action), where action is
    # the action chosen by the main network.
    def find_target_value(self, reward, next_state):
        nxt = self.target_net.feedforward(next_state)
        argmaxIdx = np.argmax(self.main_net.feedforward(next_state))
        x = nxt[argmaxIdx]
        # x = reward + self.discount_factor \
        #        * self.target_net.feedforward(next_state)[np.argmax(self.main_net.feedforward(next_state))]
        
        return x



def load(fname):
    try:
        with open(fname, 'rb') as f:
            return pickle.load(f)

    except OSError:
        print("ERROR: File not found.")
        return