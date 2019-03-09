import numpy as np
import torch
from torch import nn
from torch import optim
import shutil
import collections
import gym
import argparse
from tensorboardX import SummaryWriter


"""
Personal implementation of A2C on Cartpole based on Pong implementation (taken from the book "DRL:hands on"). Goal:
verify understanding and validate intuitions.
Experiments: the code seems to work, convergence is achieved in a reasonable amount of steps.
"""

# Global variables
LEARNING_RATE = 0.001
NB_ENVS = 4
BATCH_SIZE = 4  # number of rollouts per batch
ROLLOUT_HORIZON = 10  # length of rollouts
REWARD_THRESHOLD = 195
GAMMA = 0.99
ENTROPY_REG = 0.01


class CartPoleA2C(nn.Module):
    def __init__(self, input_size, nb_actions):
        super(CartPoleA2C, self).__init__()

        self.policy_net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, nb_actions)
        )

        self.value_net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.policy_net(x), self.value_net(x)


Batch = collections.namedtuple('Batch', field_names=['states_batch', 'actions_batch', 'qvals_batch',
                                                     'nb_samples_batch', 'nb_rollouts'])


class Agent:
    def __init__(self, policy_net, value_net, gamma):
        self.gamma = gamma
        self.policy_net = policy_net
        self.value_net = value_net

    def compute_qval(self, rewards, last_state, done):  # computes empirical Q-values of a sequence of rewards
        qvals = []
        if done:
            qval = 0.  # if the episode is done, set the value function to 0 (very important!)
        else:
            qval = self.value_net(torch.FloatTensor([last_state])).item()

        for r in reversed(rewards):
            qval *= self.gamma
            qval += r
            qvals.append(qval)

        return list(reversed(qvals))

    def iterate_batches(self, envs, tracker, max_batch_size=1, rollout_horizon=1, device=torch.device('cpu')):
        nb_actions = envs[0].action_space.n
        nb_envs = len(envs)

        # Counters
        calls_env = 0  # counts the total number of calls to env.step()
        episodes = 0  # counts the number of episodes
        nb_rollouts_batch = 0  # counts number of rollouts in batch

        # Batch samples
        states_batch = []
        actions_batch = []
        qvals_batch = []

        current_states_envs = [env.reset() for env in envs]  # current states of all environments
        reward_episode_envs = [0. for _ in
                               range(nb_envs)]  # keeps track of cumulative reward of episode in all environments

        env_id = 0  # environment id

        task_solved = False

        while True:  # Loop over rollouts while task not solved
            states_rollout = []
            actions_rollout = []
            rewards_rollout = []
            state = current_states_envs[env_id]
            nb_samples_batch = 0  # number of calls to env.step() in batch
            env = envs[env_id]
            for rollout_step in range(rollout_horizon):  # for every rollout, loop over steps
                # Compute action probabilities
                logit_action = self.policy_net(torch.FloatTensor([state]).to(device))
                actions_proba = nn.functional.softmax(logit_action, dim=1).data.numpy()[0]

                # Sample action, execute action and obtain new sample from the environment
                action = np.random.choice(nb_actions, p=actions_proba)
                next_state, reward, done, _ = env.step(action)
                states_rollout.append(state)
                actions_rollout.append(action)
                rewards_rollout.append(reward)
                reward_episode_envs[env_id] += reward
                calls_env += 1  # update number of calls to ebv.step()
                nb_samples_batch += 1
                state = next_state  # update current state

                if done:  # Check if episode has ended
                    state = env.reset()
                    episodes += 1  # update number of episodes
                    task_solved = tracker.check_reward(reward_episode_envs[env_id], calls_env)
                    reward_episode_envs[env_id] = 0.  # reset cumulative reward of episode for this environment
                    break

            if task_solved:  # if task is solved, stop!
                break

            # Update batch with new rollout(s):
            states_batch.append(states_rollout)
            actions_batch.append(actions_rollout)
            qvals_batch.append(self.compute_qval(rewards_rollout, next_state, done))
            nb_rollouts_batch += 1  # update number of samples in batch

            current_states_envs[env_id] = state  # update current state of environment

            # Update environment id
            env_id += 1
            env_id %= nb_envs

            if nb_rollouts_batch < max_batch_size:  # Check if maximum batch size has been reached, if not continue
                continue

            yield Batch(states_batch=states_batch, actions_batch=actions_batch,
                        qvals_batch=qvals_batch, nb_samples_batch=nb_samples_batch,
                        nb_rollouts=nb_rollouts_batch)  # otherwise return batch data

            # Clear all batch samples
            nb_rollouts_batch = 0
            states_batch.clear()
            actions_batch.clear()
            qvals_batch.clear()


class Tracker:  # Tracks all values of interest and check termination, to be used with "with"
    def __init__(self, tx_folder_name, reward_threshold):
        self.tx_folder_name = tx_folder_name
        self.reward_threshold = reward_threshold

    def __enter__(self):
        self.reward_episodes = []
        try:
            shutil.rmtree(self.tx_folder_name)
            self.writer = SummaryWriter(self.tx_folder_name)
        except:
            self.writer = SummaryWriter(self.tx_folder_name)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.writer.close()

    def check_reward(self, reward_episode, step):
        self.reward_episodes.append(reward_episode)
        mean_reward_episodes_100 = np.mean(self.reward_episodes[-100:])
        print('%d episodes done (%d steps), reward of last episode: %.2f, average reward of last 100 episodes: %.2f'
              %(len(self.reward_episodes), step, reward_episode, mean_reward_episodes_100))
        self.writer.add_scalar('reward_episode', reward_episode, step)
        self.writer.add_scalar('reward_episode_mean100', mean_reward_episodes_100, step)
        if mean_reward_episodes_100 > self.reward_threshold:
            print('Task solved in %d episodes and %d steps.' %(len(self.reward_episodes), step))
            return True  # Task solved
        return False  # Task not solved yet

    def track(self, name, value, step):
        self.writer.add_scalar(name, value, step)


def compute_loss(states_batch, actions_batch, qvals_batch, net):
    loss_policy = 0.
    loss_value = 0.

    test =list(zip(states_batch, actions_batch, qvals_batch))

    for (states_rollout, actions_rollout, qvals_rollout) in zip(states_batch, actions_batch, qvals_batch):
        len_rollout = len(states_rollout)

        # Convert to pytorch tensor
        states_rollout, actions_rollout, qvals_rollout = torch.FloatTensor(states_rollout),\
                                                         torch.LongTensor(actions_rollout),\
                                                         torch.FloatTensor(qvals_rollout)

        policy_logits, values = net(states_rollout)  # predictions from network

        discounts = torch.FloatTensor([GAMMA ** i for i in range(len_rollout)])

        # Loss of the policy (policy gradient theorem)
        log_proba_actions = nn.functional.log_softmax(policy_logits, dim=1)
        log_policy = log_proba_actions[range(len_rollout), actions_rollout]
        cum_log_proba_actions = (log_policy * discounts).cumsum(dim=0)  # "Double some" of policy gradient handled correctly!

        adv_log_policy = -(qvals_rollout - values.detach()) * cum_log_proba_actions  # Don't forget to detach fixed tensors
        loss_policy += adv_log_policy.mean()

        # Loss of the value function
        loss_value += nn.functional.mse_loss(qvals_rollout, values)

        # Entropy regularization (N.B: we want to max entropy but min loss)
        proba_actions = nn.functional.softmax(policy_logits, dim=1)
        entropy_loss = ENTROPY_REG * (proba_actions * log_proba_actions).sum(dim=1).mean()
        
    return loss_policy + loss_value + entropy_loss  # Sum all losses and update gradient



if __name__=='__main__':
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true', default=False, help='Enable cuda (disabled by default)')
    args = parser.parse_args()
    device = torch.device('cuda' if args.cuda else 'cpu')

    # Use several environments to reduce correlation between samples
    envs = [gym.make('CartPole-v0') for _ in range(NB_ENVS)]

    # Create policy and value network
    net = CartPoleA2C(envs[0].observation_space.shape[0], envs[0].action_space.n).to(device)

    # print(len(list(net.parameters())))  # Number of layers
    # print(sum(p.numel() for p in net.parameters()))  # Number of parameters
    # print(sum(p.numel() for p in net.parameters() if p.requires_grad))  # Number of parameters

    agent = Agent(lambda x: net(x)[0], lambda x: net(x)[1], GAMMA)  # agent

    optimizer = optim.Adam(params=net.parameters(), lr=LEARNING_RATE)

    tx_folder_name = 'runs-cartpole-a2c'

    with Tracker(tx_folder_name, REWARD_THRESHOLD) as tracker:
        for i, batch in enumerate(agent.iterate_batches(envs, tracker, max_batch_size=BATCH_SIZE,
                                                        rollout_horizon=ROLLOUT_HORIZON, device=device)):
            # # Gather batch samples
            # states_batch_t = torch.FloatTensor(batch.states_batch)
            # actions_batch_t = torch.LongTensor(batch.actions_batch)
            # qvals_batch_t = torch.FloatTensor(batch.qvals_batch)

            optimizer.zero_grad()  # Don't forget to zero the gradients!

            # policy_logits, values = net(states_batch_t)  # predictions from network
            #
            # # Loss of the policy (policy gradient theorem): does not account for the "double sum"
            # log_proba_actions = nn.functional.log_softmax(policy_logits, dim=1)
            # adv_log_pol = - (qvals_batch_t - values.detach()) * log_proba_actions[range(batch.batch_size), actions_batch_t]  # Don't forget to detach fixed tensors
            # loss_policy = adv_log_pol.mean()
            #
            # # Loss of the value function
            # loss_value = nn.functional.mse_loss(qvals_batch_t, values)

            loss = compute_loss(batch.states_batch, batch.actions_batch, batch.qvals_batch, net)
            loss.backward()
            optimizer.step()

# TODO: finish implementing + run tests
