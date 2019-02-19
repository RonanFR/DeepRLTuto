import numpy as np
import gym
import torch
import cv2 as cv
# import ptan
import time
from tensorboardX import SummaryWriter


# Main packages/libraries to install (using pip, conda or pip in conda):
#  - Gym (RL library by OpenAI)
#  - OpenCV (computer vision)
#  - PyTorch (Deep Learning)
#  - ptan (developped by Maxim Lapan, author of "Deep RL Hands-on")


# Test Gym
dtype = np.float32
env = gym.make('CartPole-v0')
env.reset()
for _ in range(100):
    # env.render()
    time.sleep(0.01)
    env.step(env.action_space.sample()) # take a random action
env.close()

# Test Pytorch
x = torch.rand(5, 3)
print(x)
print(torch.cuda.is_available())

# Test OpenCV
img = cv.imread('pictures/img1.jpg')
px = img[1,1]
print( px )

# TensorboardX
writer = SummaryWriter('test_summaryWriter')
writer.close()