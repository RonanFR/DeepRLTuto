import gym
import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
import shutil
from collections import namedtuple


# Following code implemented based on what we learned on the CartPole example
# As explained in the book, this example is failing to run if implemented like in CartPole (the reward signal is not
# informative enough and getting the reward requires too much time and so most episodes are failing).

# Constants
BATCH_SIZE = 100
HIDDEN_SIZE = 128
PERCENTILE = 70

# We implement the "one-hot encoding" trick: discrete inputs in {1,...,n} are represented by an n-dimensional vector which
# takes value 0 everywhere except at the coordinate it represents where it takes value 1
class OneHotEncoding(gym.ObservationWrapper):

    def __init__(self, env):
        super(OneHotEncoding, self).__init__(env)
        assert isinstance(env.observation_space, gym.spaces.Discrete)
        self.nb_states = env.observation_space.n
        self.observation_space = gym.spaces.Box(0.0, 1.0, (self.nb_states, ), dtype=np.float32)

    def observation(self, observation):
        vec = np.copy(self.observation_space.low)
        vec[observation] = 1.0
        return vec

# Neural Net
class Net(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.net(x)

# define new namedtuple for episodes
Episode = namedtuple('Episode', field_names = ['reward','steps'])
EpisodeStep = namedtuple('EpisodeStep', field_names = ['observation', 'action'])

# iterate over batches
def iterate_batches(env, net, batch_size):
    batch = []
    nb_episodes = 0
    cumulative_reward = 0.
    episode_steps = []
    obs = env.reset()
    sm = nn.Softmax(dim = 1)
    while True:
        act_val = net(torch.tensor([obs]))
        act_proba = sm(act_val)
        act = np.random.choice(env.action_space.n, p=act_proba.data.numpy()[0])
        new_obs, reward, is_done, _ = env.step(act)
        cumulative_reward += reward
        episode_steps.append(EpisodeStep(observation=obs, action=act))
        if is_done:
            batch.append(Episode(reward=cumulative_reward, steps=episode_steps))
            nb_episodes += 1
            cumulative_reward = 0.
            episode_steps = []
            new_obs = env.reset()
            if nb_episodes == batch_size:
                yield batch
                batch = []
                nb_episodes = 0
        obs = new_obs

# Filter batch for cross-entropy loss
def filter_batch(batch_sample, percentile):
    rewards = [episode.reward for episode in batch_sample]
    # rewards = [episode.reward / len(episode.steps) for episode in batch_sample]  # we use the gain instead of the cumulative reward to improve performances
    reward_bound = np.percentile(rewards, percentile)
    reward_mean = np.mean(rewards)
    batch_actions = []
    batch_observation = []
    for episode in batch_sample:
        if episode.reward >= reward_bound:
            actions = [step.action for step in episode.steps]
            observations = [step.observation for step in episode.steps]
            batch_actions += actions
            batch_observation += observations
    return torch.tensor(batch_actions), torch.tensor(batch_observation), reward_bound, reward_mean


if __name__ == '__main__':
    env = gym.make('FrozenLake-v0')
    wrapped_env = OneHotEncoding(env)  # change the observation space of Frozen Lake to be able to pass it to a Neural Net
    net = Net(wrapped_env.observation_space.shape[0], HIDDEN_SIZE, wrapped_env.action_space.n)
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    try:
        shutil.rmtree('runs_frozenlake')
        writer = SummaryWriter('runs_frozenlake')
    except:
        writer = SummaryWriter('runs_frozenlake')
    for nb_iter, batch_sample in enumerate(iterate_batches(wrapped_env, net, BATCH_SIZE)):
        optimizer.zero_grad()
        actions, observations, reward_bound, reward_mean = filter_batch(batch_sample, PERCENTILE)
        loss_val = loss(net(observations), actions)
        loss_val.backward()
        optimizer.step()
        writer.add_scalar('reward_mean', reward_mean, nb_iter)
        writer.add_scalar('reward_bound', reward_bound, nb_iter)
        writer.add_scalar('loss', loss_val.item(), nb_iter)
        print('%d: loss=%.3f, reward_mean=%.1f, reward_bound=%.1f' % (nb_iter, loss_val.item(), reward_bound, reward_mean))
        if reward_mean >199:
            print('Task solved!')
            break
    writer.close()



