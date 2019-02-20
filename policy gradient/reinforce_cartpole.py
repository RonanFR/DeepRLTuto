import gym
import numpy as np
import torch
import ptan
from tensorboardX import SummaryWriter
import shutil
from torch import optim
from torch import nn


GAMMA = 0.99
LEARNING_RATE = 0.001
ENTROPY_BETA = 0.01
BATCH_SIZE = 8
REWARD_STEPS = 10

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

    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=GAMMA, steps_count=REWARD_STEPS)

    optimizer = optim.Adam(params=net.parameters(), lr=LEARNING_RATE)

    total_rewards = []
    done_episodes = 0

    batch_states, batch_actions, batch_qvals = [], [], []

    reward_sum = 0.0
    batch_scales = []

    for i, exp in enumerate(exp_source):
        reward_sum += exp.reward
        baseline = reward_sum/(i+1)
        writer.add_scalar('baseline', baseline, i)
        batch_states.append(exp.state)
        batch_actions.append(int(exp.action))
        batch_scales.append(exp.reward - baseline)

        # handle new rewards
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

        if len(batch_states) < BATCH_SIZE:
            continue

        optimizer.zero_grad()
        states_v = torch.FloatTensor(batch_states)
        batch_actions_t = torch.LongTensor(batch_actions)
        batch_scale_v = torch.FloatTensor(batch_scales)

        logits_v = net(states_v)
        log_prob_v = nn.functional.log_softmax(logits_v, dim=1)
        log_prob_actions_v = batch_scale_v * log_prob_v[range(len(batch_states)), batch_actions_t]
        loss_policy_v = -log_prob_actions_v.mean()

        prob_v = nn.functional.softmax(logits_v, dim=1)
        entropy_v = -(prob_v * log_prob_v).sum(dim=1).mean()
        entropy_loss_v = -ENTROPY_BETA * entropy_v

        loss_v = loss_policy_v + entropy_loss_v

        loss_v.backward()
        optimizer.step()

        # compute KL-div to monitor learning process
        new_logits_v = net(states_v)
        new_prob_v = nn.functional.softmax(new_logits_v, dim=1)
        kl_div_v = -((new_prob_v / prob_v).log() * prob_v).sum(dim=1).mean()
        writer.add_scalar('KL', kl_div_v.item(), i)

        writer.add_scalar("entropy", entropy_v.item(), i)
        writer.add_scalar("loss_entropy", entropy_loss_v.item(), i)
        writer.add_scalar("loss_policy", loss_policy_v.item(), i)
        writer.add_scalar("loss_total", loss_v.item(), i)

        batch_states.clear()
        batch_actions.clear()
        batch_scales.clear()

    writer.close()















