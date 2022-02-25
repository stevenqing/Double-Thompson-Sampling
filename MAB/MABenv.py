#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 12:05:47 2021

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
from DTS import DTS
import matplotlib.pyplot as plt



class MAB(object):
    def __init__(self,**config):
        self.original_distribution = config['original_distribution']
        self.mu = config['mu']
        self.var = config['var']
        self.alpha = config['alpha']
        self.beta = config['beta']
        self.total_arms_number = config['total_arms_number']
        self.policy_arms_number = config['policy_arms_number']
        self.step = config['step']
        self.hyper_distribution_parameter = config['hyper_distribution_parameter']
        self.stochastic_game = config['stochastic_game']
    def generate_totalarms(self):
        if self.stochastic_game == False:
            random.seed(self.hyper_distribution_parameter)
        self.total_arms = []
        if self.original_distribution == 'Gaussian':
            hyper_mu = [np.random.normal(self.mu, self.var)*10 for _ in range(self.total_arms_number)]
            hyper_var = [np.random.normal(self.mu, self.var) for _ in range(self.total_arms_number)]
            
            for i in range(self.total_arms_number):
                self.total_arms.append(np.random.normal(hyper_mu[i],hyper_var[i]))
        
        elif self.original_distribution == 'Beta':
            hyper_alpha = [random.betavariate(self.alpha, self.beta) for _ in range(self.total_arms_number)]
            hyper_beta = [random.betavariate(self.alpha, self.beta) for _ in range(self.total_arms_number)]
            
            for i in range(self.total_arms_number):
                self.total_arms.append(random.betavariate(hyper_alpha[i], hyper_beta[i]))
        
        elif self.original_distribution == 'Normal':
            self.total_arms = [random.uniform(0, self.total_arms_number) for _ in range(self.total_arms_number)]
        
        elif self.original_distribution == 'Bernoulli':
            self.total_arms = [random.randint(0, 1) for _ in range(self.total_arms_number)]
            
        else:
            self.total_arms = [np.random.normal(1, 1) for _ in range(self.total_arms_number)]
        
        if min(self.total_arms) < 0:
            self.total_arms = [i-min(self.total_arms) for i in self.total_arms] #令所有reward大于0
        
    def Initialize(self):
        self.generate_totalarms()
        arms = self.total_arms
        count = [x for x in range(len(self.total_arms))]
        zipped = zip(arms,count)
        sort_zipped = sorted(zipped,key = lambda x:x[0],reverse=True)
        result = zip(*sort_zipped)
        x,y = [list(x) for x in result]
        return y
    
    def RegretCal(self,agent,config):
        agent = agent(**config)
        agent.initialize()
        regret = []
        tmp_regret = 0
        for _ in range(self.step):
            n = agent.select_arm()
            # self.generate_totalarms()
            agent.update(n,self.total_arms[n])
            agent_move,max_move = agent.select_arm(),self.total_arms.index(max(self.total_arms))
            tmp_regret += self.total_arms[max_move] - self.total_arms[agent_move] 
            regret.append(tmp_regret)
        arm = agent.select_arm()
        return regret,arm
    
    
    
    def Greedy(self,config):
        agent = EpsilonGreedy(**config)
        agent.initialize()
        regret = []
        tmp_regret = 0
        for _ in range(self.step):
            n = agent.select_arm()
            # self.generate_totalarms()
            agent.update(n,self.total_arms[n])
            agent_move,max_move = agent.select_arm(),self.total_arms.index(max(self.total_arms))
            tmp_regret += self.total_arms[max_move] - self.total_arms[agent_move] 
            regret.append(tmp_regret)
        arm = agent.select_arm()
        return regret,arm
    
    def UCB(self,config):
        agent = UCB1(**config)
        agent.initialize()
        regret = []
        tmp_regret = 0
        for _ in range(self.step):
            n = agent.select_arm()
            agent.update(n,self.total_arms[n])
            agent_move,max_move = agent.select_arm(),self.total_arms.index(max(self.total_arms))
            tmp_regret += self.total_arms[max_move] - self.total_arms[agent_move] 
            regret.append(tmp_regret)
        arm = agent.select_arm()
        return regret,arm
    
    def SoftMax(self,config):
        agent = Boltzmann(**config)
        agent.initialize()
        regret = []
        tmp_regret = 0
        for _ in range(self.step):
            n = agent.select_arm()
            agent.update(n,self.total_arms[n])
            agent_move,max_move = agent.select_arm(),self.total_arms.index(max(self.total_arms))
            tmp_regret += self.total_arms[max_move] - self.total_arms[agent_move] 
            regret.append(tmp_regret)
        arm = agent.select_arm()
        return regret,arm
            
    def TS(self,config):
        agent = ThompsonSampling(**config)
        agent.initialize()
        regret = []
        tmp_regret = 0
        for _ in range(self.step):
            n = agent.select_arm()
            agent.update(n,self.total_arms[n])
            agent_move,max_move = agent.select_arm(),self.total_arms.index(max(self.total_arms))
            tmp_regret += self.total_arms[max_move] - self.total_arms[agent_move] 
            regret.append(tmp_regret)
        arm = agent.select_arm()
        return regret,arm
        
    def BayesianUCB(self,config):
        agent = BayesianUCB(**config)
        agent.initialize()
        regret = []
        tmp_regret = 0
        for i in range(self.step):
            n = agent.select_arm()
            agent.update(n,self.total_arms[n])
            agent_move,max_move = agent.select_arm(),self.total_arms.index(max(self.total_arms))
            tmp_regret += self.total_arms[max_move] - self.total_arms[agent_move] 
            regret.append(tmp_regret)
        arm = agent.select_arm()
        return regret,arm

    def SWTS(self,config):
        agent = SWTS(**config)
        agent.initialize()
        regret = []
        tmp_regret = 0
        for _ in range(self.step):
            n = agent.select_arm()
            agent.update(n,self.total_arms[n])
            agent_move,max_move = agent.select_arm(),self.total_arms.index(max(self.total_arms))
            tmp_regret += self.total_arms[max_move] - self.total_arms[agent_move] 
            regret.append(tmp_regret)
        arm = agent.select_arm()
        return regret,arm
    
    def DTS(self,config):
        agent = DTS(**config)
        agent.initialize()
        regret = []
        tmp_regret = 0
        for _ in range(self.step):
            n = agent.select_arm()
            agent.update(n,self.total_arms[n])
            agent_move,max_move = agent.select_arm(),self.total_arms.index(max(self.total_arms))
            tmp_regret += self.total_arms[max_move] - self.total_arms[agent_move] 
            regret.append(tmp_regret)
        arm = agent.select_arm()
        return regret,arm
        
if __name__ == "__main__":

    hyper_config = {
        "original_distribution" : 'Beta',
        "mu" : 10,
        "var" : 15,
        "alpha" : 10,
        "beta" : 20,
        "total_arms_number" : 500,
        "policy_arms_number" : 1,
        "step" : 100,
        "hyper_distribution_parameter" : 123,
        "stochastic_game" : True
        }

    MAB_env = MAB(**hyper_config)
    total_arms = MAB_env.Initialize()
# testing the e-greedy algorithm
    greedy_config = {
        "epsilon": 1,
        "policy_arms_number": 1,
        "total_arms_number": 500,
        }
    regret_greedy,arm = MAB_env.RegretCalvhn(EpsilonGreedy, greedy_config)
    print(regret_greedy[0],regret_greedy[-1],total_arms.index(arm))
    
# testing the UCB1 algorithm
    UCB_config = {
        "policy_arms_number": 1,
        "total_arms_number": 500,
        }
    regret_UCB,arm = MAB_env.UCB(UCB_config)
    print(regret_UCB[0],regret_UCB[-1],total_arms.index(arm))    

# # testing the Boltzmann algorithm
    Boltzmann_config = {
        "tau": 20,
        "policy_arms_number": 1,
        "total_arms_number": 500,
        }
    regret_Boltzmann,arm = MAB_env.SoftMax(Boltzmann_config)
    print(regret_Boltzmann[0],regret_Boltzmann[-1],total_arms.index(arm))
    
# # testing the TS algorithm
    TS_config = {
        "policy_arms_number": 1,
        "total_arms_number": 500,
        }
    regret_TS,arm = MAB_env.TS(TS_config)
    print(regret_TS[0],regret_TS[-1],total_arms.index(arm))

# # testing the BayesianUCB algorithm
    BayesianUCB_config = {
        "upper_bound_dev": 3,
        "policy_arms_number": 1,
        "total_arms_number": 500,
        }
    regret_BUCB,arm = MAB_env.BayesianUCB(BayesianUCB_config)
    print(regret_BUCB[0],regret_BUCB[-1],total_arms.index(arm))

# # testing the SWTS algorithm
    SWTS_config = {
        "sliding_window_size":2,
        "policy_arms_number": 1,
        "total_arms_number": 500,
        }
    regret_SWTS,arm = MAB_env.SWTS(SWTS_config)
    print(regret_SWTS[0],regret_SWTS[-1],total_arms.index(arm))
        
# # testing the DTS algorithm
    DTS_config = {
        "upper_bound_dev": 3,
        "policy_arms_number": 1,
        "total_arms_number": 500,
        }
    regret_DTS,arm = MAB_env.DTS(DTS_config)
    print(regret_DTS[0],regret_DTS[-1],total_arms.index(arm))
    
x_axis = [i for i in range(len(regret_greedy))]


sub_axix = filter(lambda x:x%200 == 0, x_axis)
plt.title('MAB')
plt.plot(x_axis, regret_greedy, color='green',linestyle='-.', label='e_greedy')
plt.plot(x_axis, regret_UCB, color='purple', label='UCB')
plt.plot(x_axis, regret_Boltzmann,  color='yellow', label='Boltzmann')
plt.plot(x_axis, regret_TS, color='blue', label='TS')
plt.plot(x_axis, regret_BUCB, color='red', label='BayesianUCB')
plt.plot(x_axis, regret_SWTS, color='black',linestyle='--', label='SWTS')
plt.plot(x_axis, regret_DTS, color='brown',label='DTS')
plt.legend() 

plt.xlabel('T')
plt.ylabel('total regret')
plt.show()

    