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
NB_ENVS = 3
BATCH_SIZE = 4
BELLMAN_HORIZON = 10
REWARD_THRESHOLD = 195
GAMMA = 0.99
ENTROPY_REG = 0.0


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


Batch = collections.namedtuple('Batch', field_names=['states_batch', 'actions_batch', 'qvals_batch'])


class Agent:
    def __init__(self, policy_net, value_net, gamma):
        self.gamma = gamma
        self.policy_net = policy_net
        self.value_net = value_net

    def compute_qval(self, rewards, last_state, done):
        if done:
            qval = 0.  # if the episode is done, set the value function to 0 (very important!)
        else:
            qval = self.value_net(torch.FloatTensor([last_state])).item()

        for r in reversed(rewards):
            qval *= self.gamma
            qval += r

        return qval

    def iterate_batches(self, envs, tracker, max_batch_size=1, bellman_horizon=1, device=torch.device('cpu')):
        nb_actions = envs[0].action_space.n
        nb_envs = len(envs)

        # Counters
        steps = 0  # counts the total number of calls to env.step()
        episodes = 0  # counts the number of episodes
        batch_size = 0  # counts number of samples in batch

        # Batch samples
        states_batch = []
        actions_batch = []
        qvals_batch = []

        current_states = [env.reset() for env in envs]  # current states of all environments
        reward_episode_envs = [0. for _ in
                               range(nb_envs)]  # keeps track of cumulative reward of episode in all environments

        env_id = 0  # environment id

        task_solved = False

        while True:  # Loop while task ot solved
            states = []
            actions = []
            rewards = []
            state = current_states[env_id]
            nb_experiences = 1
            env = envs[env_id]
            for horizon in range(bellman_horizon):
                # Compute action probabilities
                logit_action = self.policy_net(torch.FloatTensor([state]).to(device))
                actions_proba = nn.functional.softmax(logit_action, dim=1).data.numpy()[0]

                # Sample action, execute action and obtain new sample from the environment
                action = np.random.choice(nb_actions, p=actions_proba)
                next_state, reward, done, _ = env.step(action)
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                reward_episode_envs[env_id] += reward
                steps += 1  # update number of steps
                state = next_state  # update current state

                if done:  # Check if episode has ended
                    state = env.reset()
                    episodes += 1  # update number of episodes
                    task_solved = tracker.check_reward(reward_episode_envs[env_id], steps)
                    reward_episode_envs[env_id] = 0.  # reset cumulative reward of environment
                    nb_experiences = horizon + 1
                    break

            if task_solved:  # if task is solved, stop!
                break

            # Update batch with new sample(s):
            #   1) if episode is not over, just use one experience (too much bias otherwise)
            #   2) automatically generate "fake" experiences with tail of batch samples when episode is over (better
            #      sample efficiency)
            for i in range(nb_experiences):
                states_batch.append(states[i])
                actions_batch.append(actions[i])
                qvals_batch.append(self.compute_qval(rewards[i:], next_state, done))
                batch_size += 1  # update number of samples in batch

            current_states[env_id] = state

            # Update environment id
            env_id += 1
            env_id %= nb_envs

            if batch_size < max_batch_size:  # Check if maximum batch size has been reached, if not continue
                continue

            yield Batch(states_batch=states_batch, actions_batch=actions_batch,
                        qvals_batch=qvals_batch)  # otherwise return batch data

            # Clear all batch samples
            batch_size = 0
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

    agent = Agent(lambda x: net(x)[0], lambda x: net(x)[1], GAMMA)  # agent

    optimizer = optim.Adam(params=net.parameters(), lr=LEARNING_RATE)

    tx_folder_name = 'runs-cartpole-a2c'

    with Tracker(tx_folder_name, REWARD_THRESHOLD) as tracker:
        for i, batch in enumerate(agent.iterate_batches(envs, tracker, max_batch_size=BATCH_SIZE,
                                                        bellman_horizon=BELLMAN_HORIZON, device=device)):
            # Gather batch samples
            batch_size = len(batch.qvals_batch)
            states_batch_t = torch.FloatTensor(batch.states_batch)
            actions_batch_t = torch.LongTensor(batch.actions_batch)
            qvals_batch_t = torch.FloatTensor(batch.qvals_batch)

            optimizer.zero_grad()  # computation graph starts here! Don't forget to detach fixed tensors

            policy_logits, values = net(states_batch_t)  # predictions from network

            # Loss of the policy (policy gradient theorem)
            log_proba_actions = nn.functional.log_softmax(policy_logits, dim=1)
            adv_log_pol = - (qvals_batch_t - values.detach()) * log_proba_actions[range(batch_size), actions_batch_t]
            loss_policy = adv_log_pol.mean()

            # Loss of the value function
            loss_value = nn.functional.mse_loss(qvals_batch_t, values)

            # Entropy regularization (N.B: we want to max entropy but min loss)
            proba_actions = nn.functional.softmax(policy_logits, dim=1)
            entropy_loss = ENTROPY_REG * (proba_actions * log_proba_actions).sum(dim=1).mean()

            # Sum all losses and update gradient
            loss = loss_value + loss_policy + entropy_loss
            loss.backward()
            optimizer.step()

# TODO: finish implementing + run tests
