import gym
import collections
from tensorboardX import SummaryWriter
import shutil


# Constants
ENV_NAME = 'FrozenLake-v0'
GAMMA = 0.9
ALPHA = 0.2
TEST_EPISODES = 20


class Agent:
    def __init__(self):
        self.env = gym.make(ENV_NAME)
        self.state = self.env.reset()
        self.values = collections.defaultdict(float)

    def sample_env(self):
        action = self.env.action_space.sample()
        old_state = self.state
        new_state, reward, is_done, _ = self.env.step(action)
        self.state = self.env.reset() if is_done else new_state
        return old_state, action, reward, new_state

    def best_value_action(self, state):
        best_value, best_action = float('-inf'), None
        for action in range(self.env.action_space.n):
            if self.values[(state, action)] > best_value:
                best_value = self.values[(state, action)]
                best_action = action
        return best_value, best_action

    def value_update(self, old_state, action, reward, new_state):
        best_value, _ = self.best_value_action(new_state)
        self.values[(old_state, action)] = (1 - ALPHA) * self.values[(old_state, action)] \
                                            + ALPHA*(reward + GAMMA*best_value)

    def play_episode(self, env):
        total_reward = 0.
        state = env.reset()
        while True:
            _, action = self.best_value_action(state)
            new_state, reward, is_done, _ = env.step(action)
            total_reward += reward
            if is_done:
                break
            state = new_state
        return total_reward


if __name__ == '__main__':
    test_env = gym.make(ENV_NAME)
    agent = Agent()
    try:
        shutil.rmtree('runs_tabular')
        writer = SummaryWriter('runs_tabular')
    except:
        writer =SummaryWriter('runs_tabular')

    iter_nb = 0
    best_reward = 0.
    while True:
        iter_nb += 1
        x, a, r, y = agent.sample_env()
        agent.value_update(x, a, r, y)

        reward = 0.
        for _ in range(TEST_EPISODES):
            reward += agent.play_episode(test_env)
        reward /= TEST_EPISODES

        writer.add_scalar('reward', reward, iter_nb)
        if reward > best_reward:
            print('Best reward updated from %.3f to %.3f' %(best_reward, reward))
            best_reward = reward
        if reward > 0.8:
            print('Task solved in %.d iterations' %(iter_nb))
            break
    writer.close()