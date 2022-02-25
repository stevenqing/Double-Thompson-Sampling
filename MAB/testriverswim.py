#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 20:03:48 2021

@author: shishuqing
"""

import numpy as np
import random
from epsilon_greedy import EpsilonGreedy,ind_max
from UCB import UCB1,find_index
from Boltzmann import Boltzmann
from ThompsonSampling import ThompsonSampling
from BayesianUCB import BayesianUCB
from SWTS import SWTS
from utils import MSE


import gym


env = gym.make('RiverSwim-v0')
state,reward = env.observe(1,5) #action,state
print(reward)
start = env.reset()



