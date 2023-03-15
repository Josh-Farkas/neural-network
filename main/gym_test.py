import dqn_agent as dqn
import gymnasium as gym
import numpy as np


ENV_NAME = "CartPole-v1"
STEPS_TO_UPDATE = 100
BATCH_SIZE = 20
LEARNING_RATE = 0.00025
N_EPISODES = 1000
SEED = 50
MEMORY_SIZE = 100000

env = gym.make(ENV_NAME, render_mode="human")
#  render_mode="human"
env.reset()
# env.render()
observation, info = env.reset(seed=SEED)
# print(env.action_space)


# n_obs = env.observation_space[0].n
# n_act = env.action_space.n

agent = dqn.DQNAgent(4, 2, [24, 24], learning_rate=LEARNING_RATE, memory_size=MEMORY_SIZE)

# print(agent.main_net.get_weights())
steps = 0
# env.render()
for episode in range(N_EPISODES):
    print(f'Episode: {episode}')
    terminated = False
    truncated = False
    start_steps = 0
    while not (terminated or truncated):
        steps += 1
        action = agent.choose_action(observation)
        # print(action)
        next_observation, reward, terminated, truncated, info = env.step(np.array(action))
        # reward = reward if not terminated else -10*reward
        # print(reward)

        agent.store_memory(observation, action, reward, next_observation, terminated or truncated)
        observation = next_observation
        # episode_reward += reward
        # if steps % 4 == 0:
        agent.train(BATCH_SIZE)

        if terminated or truncated:
            print(f'Episode {episode} lasted for {steps-start_steps} steps')
            # print(episode_reward)
            if steps >= STEPS_TO_UPDATE:
                agent.update_target()
                steps = 0

    observation, info = env.reset()
    # if episode == 900:
    #     env.render()

print(agent.main_net.get_weights())
env.close()