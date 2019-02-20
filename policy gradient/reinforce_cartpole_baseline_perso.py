import gym
import torch
from torch import nn
from torch import optim
import collections
import shutil
import numpy as np
from tensorboardX import SummaryWriter


"""
This is a personal implementation of REINFORCE for CartPole not using the PTAN library to verify understanding of the 
algorithm: how data are processed, how the policy is learned, influence of parameters, getting used to debugging, etc.
"""

ENV = "CartPole-v0"
HIDDEN_LAYERS = 128
BELLMAN_HORIZON = 10  # BELLMAN_HORIZON = 200 corresponds to the full horizon
BATCH_SIZE = 4
LEARNING_RATE = 0.001
ENTROPY_REG = 0.01
GAMMA = 0.99


# Define neural network
class policy_nn_single_layer(nn.Module):
    def __init__(self, nb_inputs, nb_hidden_layers, nb_outputs):
        super(policy_nn_single_layer, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(nb_inputs, nb_hidden_layers),
            nn.ReLU(),
            nn.Linear(nb_hidden_layers, nb_outputs)
        )

    def forward(self, x):
        return self.net(x)


# Global variables
batch_experience = collections.namedtuple('Batch_experience', field_names=['states', 'actions', 'rewards'])
nb_episodes = 0
nb_samples = 0
total_rewards = []


def compute_qval(rewards, gamma):
    qval = 0.0
    qvals = []
    for r in reversed(rewards):
        qval *= gamma
        qval += r
        qvals.append(qval)
    return list(reversed(qvals))


def iterate_batch(env, net, batch_size=1, bellman_horizon=1, gamma=GAMMA):
    global nb_episodes, nb_samples, total_rewards
    batch_states = []
    batch_actions = []
    batch_rewards = []
    nb_experiences = 0
    obs = env.reset()
    nb_steps_experience = 0
    rewards_experience = []
    obs_experience = []
    actions_experience = []
    cum_reward_episode = 0.
    total_cum_discounted_reward = 0.
    total_nb_experiencess = 0
    while True:
        action_probas = nn.functional.softmax(net(torch.FloatTensor([obs])), dim=1)
        action = np.random.choice(env.action_space.n, p=action_probas.data.numpy()[0])
        next_obs, reward, done, _ = env.step(action)
        nb_steps_experience += 1
        nb_samples += 1
        cum_reward_episode += reward
        rewards_experience.append(reward)
        obs_experience.append(obs)
        actions_experience.append(action)
        if nb_steps_experience < bellman_horizon and not done:
            obs = next_obs
            continue
        if done:
            next_obs = env.reset()
            nb_episodes += 1
            total_rewards.append(cum_reward_episode)
            cum_reward_episode = 0
            n = len(rewards_experience)  # generate "fake" experiences with tail of experience when episode is over (better sample efficiency)
        else:
            n = 1  # if episode is not over, just use one experience (too much bias otherwise)
        for i in range(n):
            qval_experience = compute_qval(rewards_experience, gamma)[i]
            total_cum_discounted_reward += qval_experience
            total_nb_experiencess += 1
            baseline = total_cum_discounted_reward/total_nb_experiencess
            batch_states.append(obs_experience[i])
            batch_actions.append(actions_experience[i])
            batch_rewards.append(qval_experience - baseline)
        nb_experiences += 1  # "fake" experiences are not counted
        obs = next_obs
        nb_steps_experience = 0
        rewards_experience.clear()
        obs_experience.clear()
        actions_experience.clear()
        if nb_experiences < batch_size:
            continue
        yield batch_experience(states=batch_states, actions=batch_actions, rewards=batch_rewards)
        batch_states.clear()
        batch_actions.clear()
        batch_rewards.clear()
        nb_experiences = 0


if __name__ == "__main__":
    env = gym.make(ENV)
    net = policy_nn_single_layer(env.observation_space.shape[0], HIDDEN_LAYERS, env.action_space.n)

    tx_folder_name = 'runs_pg_'+ENV
    try:  # erase existing folder if applicable, then create new folder to store SummaryWriter
        shutil.rmtree(tx_folder_name)
        writer = SummaryWriter(tx_folder_name)
    except:
        writer = SummaryWriter(tx_folder_name)

    optimizer = optim.Adam(params=net.parameters(), lr=LEARNING_RATE)

    for _, batch_exp in enumerate(iterate_batch(env, net, batch_size=BATCH_SIZE, bellman_horizon=BELLMAN_HORIZON, gamma=1)):  # Sample environment
        batch_states_t = torch.FloatTensor(batch_exp.states)
        batch_actions_t = torch.LongTensor(batch_exp.actions)
        batch_rewards_t = torch.FloatTensor(batch_exp.rewards)

        optimizer.zero_grad()  # very important (defines from where to start the graph for computing the gradient)
        log_probas_actions_t = nn.functional.log_softmax(net(batch_states_t), dim=1)
        log_actions_t = log_probas_actions_t[range(len(batch_exp.actions)), batch_actions_t]
        losses = - batch_rewards_t * log_actions_t
        loss_policy = losses.mean()  # compute loss policy gradient
        probas_t = nn.functional.softmax(net(batch_states_t), dim=1)
        entropy_policy = - probas_t * log_probas_actions_t
        loss_entropy = - ENTROPY_REG * entropy_policy.sum(dim=1).mean()  # compute loss entropy
        loss = loss_policy + loss_entropy

        # Gradient update
        loss.backward()
        optimizer.step()

        reward_100 = np.mean(total_rewards[-100:])
        print('%d: reward: %6.2f, reward_100: %6.2f, episodes: %d' % (nb_samples, total_rewards[-1], reward_100, nb_episodes))

        writer.add_scalar("reward", total_rewards[-1], nb_samples)
        writer.add_scalar("reward_100", reward_100, nb_samples)
        writer.add_scalar("episodes", nb_episodes, nb_samples)

        if reward_100 > 195:
            print('Task solved in %d episodes' % (nb_episodes))
            break

    writer.close()  # Don't forget to close the SummaryWriter









