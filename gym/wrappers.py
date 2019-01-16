import gym
import numpy as np


# Create an epsilon-greedy agent using gym wrappers
class EpsilonGreedyAgent(gym.ActionWrapper):
    def __init__(self, env, epsilon = 0.1):
        super(EpsilonGreedyAgent, self).__init__(env)
        self.epsilon = epsilon
    def action(self, action):
        if np.random.binomial(1,self.epsilon,1):
            print('Exploratory action')
            return env.action_space.sample()
        else:
            return action

class RewardShapingAgent(gym.RewardWrapper):
    def __init__(self, env, reward_amplicficator = 10):
        super(RewardShapingAgent, self).__init__(env)
        self.reward_amplicficator = reward_amplicficator
    def reward(self, reward):
        return self.reward_amplicficator*reward

env = gym.make('CartPole-v0')
epsilonGreedyEnv = EpsilonGreedyAgent(env,0.5)
rewardShapingEnv = RewardShapingAgent(env, 20)

epsilonGreedyEnv.reset()
nb_steps = 0
cum_rewards = 0.
done = False
while not done:
    _, reward, done, _ = epsilonGreedyEnv.step(0)
    cum_rewards += reward
    nb_steps += 1
print('Total number of steps: '+str(nb_steps))
print('Total cumulated rewards: '+str(cum_rewards))

rewardShapingEnv.reset()
nb_steps = 0
cum_rewards = 0.
done = False
while not done:
    _, reward, done, _ = rewardShapingEnv.step(0)
    cum_rewards += reward
    nb_steps += 1
print('Total number of steps: '+str(nb_steps))
print('Total cumulated rewards: '+str(cum_rewards))

