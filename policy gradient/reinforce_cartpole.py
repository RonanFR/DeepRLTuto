import gym
import numpy as np
import torch
import ptan
from tensorboardX import SummaryWriter
import shutil
from torch import optim
from torch import tensor
from torch import nn


GAMMA = 0.99
LEARNING_RATE = 0.01
EPISODES_TO_TRAIN = 4

class PGN(nn.Module):  # Policy network
    def __init__(self, input_size, nb_actions):
        super(PGN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, nb_actions)
        )

    def forward(self, x):
        return self.net(x)

def calc_qvals(rewards):
    res = []
    sum_r = 0.0
    for r in rewards:
        sum_r *= GAMMA
        sum_r += r
        res.append(sum_r)
    return list(reversed(res))

if __name__=="__main__":
    env = gym.make('CartPole-v0')
    tx_folder_name = 'runs_cartpole_policy_gradient'

    try:
        shutil.rmtree(tx_folder_name)
        writer = SummaryWriter(tx_folder_name)
    except:
        writer = SummaryWriter(tx_folder_name)

    net = PGN(env.observation_space.shape[0], env.action_space.n)

    agent = ptan.agent.PolicyAgent(net, preprocessor=ptan.agent.float32_preprocessor, apply_softmax=True)

    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=GAMMA)

    optimizer = optim.Adam(params=net.parameters(), lr=LEARNING_RATE)

    total_rewards = []
    done_episodes = 0

    batch_episodes = 0
    cur_rewards = []
    batch_states, batch_actions, batch_qvals = [], [], []

    for i, exp in enumerate(exp_source):
        batch_states.append(exp.state)
        batch_actions.append(int(exp.action))
        cur_rewards.append(exp.reward)

        if exp.last_state is None:
            batch_qvals.extend(calc_qvals(cur_rewards))
            cur_rewards.clear()
            batch_episodes += 1

        new_rewards = exp_source.pop_total_rewards()
        if new_rewards:
            done_episodes += 1
            reward = new_rewards[0]
            total_rewards.append(reward)
            mean_rewards = float(np.mean(total_rewards[-100:]))
            print("%d: reward: %6.2f, mean_100: %6.2f, episodes: %d" % (i, reward, mean_rewards, done_episodes))
            writer.add_scalar("reward", reward, i)
            writer.add_scalar("reward_100", mean_rewards, i)
            writer.add_scalar("episodes", done_episodes, i)
            if mean_rewards>195:
                print("Solved in %d steps and %d episodes!!!" % (i, done_episodes))
                break

        if batch_episodes < EPISODES_TO_TRAIN:
            continue

        optimizer.zero_grad()
        states_v = torch.FloatTensor(batch_states)
        batch_actions_t = torch.LongTensor(batch_actions)
        batch_qvals_v = torch.FloatTensor(batch_qvals)

        logits_v = net(states_v)
        log_prob_v = nn.functional.log_softmax(logits_v, dim=1)
        log_prob_actions_v = batch_qvals_v * log_prob_v[range(len(batch_states)), batch_actions_t]
        loss_v = -log_prob_actions_v.mean()

        loss_v.backward()
        optimizer.step()

        batch_episodes = 0
        batch_states.clear()
        batch_actions.clear()
        batch_qvals.clear()

    writer.close()















