import gym


env = gym.make('CartPole-v0')
obs = env.reset()
print(obs)

# Try the following commands in console (first copy the above code)
env.action_space
env.observation_space
env.step(0)
env.action_space.sample()
env.observation_space.sample()

# Example of a random agent
env.reset()
cum_reward = 0.
nb_steps = 0
done = False
while not done:
    action = env.action_space.sample()
    _ , reward, done, _ = env.step(action)
    cum_reward += reward
    nb_steps += 1
print('Duration of the episode: '+str(nb_steps))
print('Cumulative reward earned: '+str(cum_reward))