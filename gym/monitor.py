import gym
import shutil


# Use Monitor to record a video of the agent
env = gym.make('CartPole-v0')
try:
    shutil.rmtree('recording_monitor')
    env = gym.wrappers.Monitor(env, 'recording_monitor')
except:
    env = gym.wrappers.Monitor(env, 'recording_monitor')

env.reset()
done = False
cum_reward = 0.
nb_steps = 0
while not done:
    _, reward, done, _ = env.step(env.action_space.sample())
    cum_reward += reward
    nb_steps += 1
print('Total number of steps: '+str(nb_steps))
print('Cumulative reward: '+str(cum_reward))
