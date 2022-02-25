# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 20:58:28 2022

@author: CILab
"""
import numpy as np
import random
from scipy.stats import beta
import math
class DTS(object):
    def __init__(self,**config):
        self.c = config['upper_bound_dev']
        self.policy_arms_number = config['policy_arms_number']
        self.total_arms_number = config['total_arms_number']
        return
    
    def initialize(self):
        '''
        Initialize count and expected value for each arm
      
        '''
        self.counts = [0 for col in range(self.total_arms_number)]
        self.alpha = [1 for col in range(self.total_arms_number)]
        self.beta = [1 for col in range(self.total_arms_number)]
        self.W = [1 for col in range(self.total_arms_number)]
        return

    def random_pick(self,l,prob):
        x = random.uniform(0,1)
        cumulative_probability=0.0
        for item,item_probability in zip(l,prob):
            cumulative_probability+=item_probability
            if x < cumulative_probability:
                break
        return item
    
    
    def select_arm(self):
        if self.policy_arms_number == 1:
            samples = [np.random.beta(self.alpha[x], self.beta[x]) for x in range(self.total_arms_number)]
            i = max(
            range(self.total_arms_number),
            key=lambda x: self.W[x]*(samples[x]) + self.c*(1-self.W[x])*(beta.std(self.alpha[x],self.beta[x]))
        )
            TS_max = max(range(self.total_arms_number),key = lambda x:(self.alpha[x] / float(self.alpha[x] + self.beta[x])))
            TS_max = self.alpha[TS_max]/float(self.alpha[TS_max]+self.beta[TS_max])
            i_value = self.alpha[i]/float(self.alpha[i]+self.beta[i])
            self.counts[i]+=1
            if TS_max != 0:
                self.W[i] = math.exp(self.counts[i]*math.log(i_value/TS_max))
            return i
        else:
            final_value = []
            for i in range(self.total_arms_number):
                expected_value = self.values[i][0]/(self.values[i][0]+self.values[i][1])
                final_value.append(expected_value)
            final_value.sort().reverse()
            return final_value[:self.policy_arms_number]
        
    
    def update(self,chosen_arm,reward):
        if self.policy_arms_number == 1:
            self.alpha[chosen_arm] += reward
            self.beta[chosen_arm] = self.counts[chosen_arm]+reward
        else:
            for i in range(self.policy_arms_number):
                self.values[chosen_arm[i]][0] += reward[i]
                self.values[chosen_arm[i]][1] = self.counts[chosen_arm[i]] - reward[i]
        return 
        
if __name__ == "__main__":
    
    arms = [1,2,3,4,5]
    config = {
        "upper_bound_dev": 3,
        "policy_arms_number": 1,
        "total_arms_number": 5,
        }
    agent = DTS(**config)
    agent.initialize()
    for _ in range(100):
        n = agent.select_arm()
        agent.update(n,arms[n])
    print(agent.alpha,agent.beta)
    n = agent.select_arm()
    print(n)
    print(agent.W)
        
        
        
        