import cv2 as cv
import gym
import gym.spaces
import numpy as np
import collections


class FireResetEnv(gym.Wrapper):
    