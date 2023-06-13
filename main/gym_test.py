import DQN
import gymnasium as gym
import numpy as np
from layers import Dense

print(gym.envs.registry.keys())
ENV_NAME = 'CartPole-v1'
STEPS_TO_UPDATE = 100
BATCH_SIZE = 32
MEMORY_SIZE = 100000
LEARNING_RATE = 0.000025
N_EPISODES = 200
SEED = None
SAVE_FILE = 'models/Agent.pkl'


env = gym.make(ENV_NAME, render_mode="human")
observation, info = env.reset(seed=SEED)

obs_shape = env.observation_space.shape
n_actions = env.action_space.n

layers = [
    Dense((24,), input_size=obs_shape, activation_func='relu'),
    Dense((24,), activation_func='relu'),
    Dense((n_actions,), activation_func='linear')
]

agent = DQN.DoubleDQNAgent(
    n_actions=n_actions, 
    layers=layers,
    epsilon_decay=.996,
    lr=LEARNING_RATE,
    memory_size=MEMORY_SIZE)

# agent = DQN.load(SAVE_FILE)

# print(agent.main_net.get_weights())
steps = 0
epsteps = 0
# env.render()
for episode in range(N_EPISODES):
    print(f'Episode: {episode}')
    terminated = False
    truncated = False
    while not (terminated or truncated):
        steps += 1
        epsteps += 1
        action = agent.act(observation)
        # print(action)
        next_observation, reward, terminated, truncated, info = env.step(np.array(action))
        # reward = reward if not terminated else reward - 10

        agent.remember(observation, action, reward, next_observation, terminated or truncated)
        observation = next_observation
        agent.train(BATCH_SIZE)

        if terminated or truncated:
            print(f'Steps: {epsteps}')
            epsteps = 0
            if steps >= STEPS_TO_UPDATE:
                agent.update_target()
                steps = 0
                

    observation, info = env.reset()


agent.save(SAVE_FILE)

env.close()