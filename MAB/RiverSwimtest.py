# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 20:44:28 2022

@author: CILab
"""

import numpy as np
import random
from epsilon_greedy import EpsilonGreedy,ind_max
from UCB import UCB1,find_index
from Boltzmann import Boltzmann
from ThompsonSampling import ThompsonSampling
from BayesianUCB import BayesianUCB
from SWTS import SWTS
from DTS import DTS
import matplotlib.pyplot as plt
import copy
from RiverSwimEnv import make_riverSwim
from DTS_riverswim import DTS_riverswim

env = make_riverSwim()
# testing the e-greedy algorithm
greedy_config = {
    "epsilon": 1,
    "policy_arms_number": 1,
    "total_arms_number": 2,
    }
regret_greedy = env.RegretCal(EpsilonGreedy,greedy_config)
print(regret_greedy[0],regret_greedy[-1])
    
# testing the UCB1 algorithm
UCB_config = {
    "policy_arms_number": 1,
    "total_arms_number": 2,
    }
regret_UCB = env.RegretCal(UCB1,UCB_config)
print(regret_UCB[0],regret_UCB[-1])    

# # testing the Boltzmann algorithm
Boltzmann_config = {
    "tau": 20,
    "policy_arms_number": 1,
    "total_arms_number": 2,
    }
regret_Boltzmann = env.RegretCal(Boltzmann,Boltzmann_config)
print(regret_Boltzmann[0],regret_Boltzmann[-1])

# # testing the TS algorithm
TS_config = {
    "policy_arms_number": 1,
    "total_arms_number": 2,
    }
regret_TS = env.RegretCal(ThompsonSampling, TS_config)
print(regret_TS[0],regret_TS[-1])

# # testing the BayesianUCB algorithm
BayesianUCB_config = {
    "upper_bound_dev": 1,
    "policy_arms_number": 1,
    "total_arms_number": 2,
    }
regret_BUCB = env.RegretCal(BayesianUCB, BayesianUCB_config)
print(regret_BUCB[0],regret_BUCB[-1])

# # testing the SWTS algorithm
SWTS_config = {
    "sliding_window_size":2,
    "policy_arms_number": 1,
    "total_arms_number": 2,
    }
regret_SWTS = env.RegretCal(SWTS, SWTS_config)
print(regret_SWTS[0],regret_SWTS[-1])
    
# # testing the DTS algorithm
# DTS_config = {
#     "upper_bound_dev": 3,
#     "policy_arms_number": 1,
#     "total_arms_number": 2,
#     }
# regret_DTS = env.RegretCal(DTS,DTS_config)
# print(regret_DTS[0],regret_DTS[-1])


DTS_Riverswim_config = {
    "policy_arms_number": 1,
    "total_arms_number": 2,
    }
regret_DTS = env.RegretCal(DTS_riverswim, DTS_Riverswim_config)
print(regret_DTS[0],regret_DTS[-1])



x_axis = [i for i in range(len(regret_greedy))]


sub_axix = filter(lambda x:x%200 == 0, x_axis)
plt.title('RiverSwim-12')
# plt.plot(x_axis, regret_greedy, color='green',linestyle='-.', label='e_greedy')
# plt.plot(x_axis, regret_UCB, color='purple', label='UCB')
# plt.plot(x_axis, regret_Boltzmann,  color='yellow', label='Boltzmann')
plt.plot(x_axis, regret_TS, color='blue', label='TS')
plt.plot(x_axis, regret_BUCB, color='red', label='BUCB')
# plt.plot(x_axis, regret_SWTS, color='black',linestyle='--', label='SWTS')
plt.plot(x_axis, regret_DTS, color='brown',label='DTS')
plt.legend() 

plt.xlabel('T')
plt.ylabel('total regret')
plt.show()
